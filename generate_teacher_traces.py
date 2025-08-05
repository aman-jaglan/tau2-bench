#!/usr/bin/env python3
"""
Generate thinking traces for all tasks using the teacher model.
Based on actual τ²-bench codebase structure.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import argparse

from tau2.data_model.message import SystemMessage
from tau2.data_model.tasks import Task
from tau2.utils.llm_utils import generate
from tau2.run import load_tasks
from tau2.registry import registry
from tau2.agent.llm_agent import LLMSoloAgent

OUTPUT_DIR = Path("data/teacher_traces")

def generate_thinking_trace_for_task(
    task: Task, 
    domain_policy: str,
    tools: List[str],
    teacher_model: str
) -> Dict:
    """Generate a thinking trace for a single task."""
    
    # Extract expected actions for validation
    expected_actions = []
    if task.evaluation_criteria and task.evaluation_criteria.actions:
        expected_actions = [
            {
                "name": action.name,
                "arguments": action.arguments,
                "requestor": action.requestor
            }
            for action in task.evaluation_criteria.actions
        ]
    
    # Create concise action string for the prompt
    action_steps = "\n".join([
        f"{i+1}. {action['name']}({json.dumps(action['arguments']) if action['arguments'] else ''})"
        for i, action in enumerate(expected_actions)
    ])
    
    trace_prompt = f"""You are an expert analyzing a technical support ticket.

TICKET: {task.ticket}

REQUIRED ACTIONS (ground truth solution):
{action_steps}

First, think deeply about this problem - analyze it thoroughly, consider all aspects, understand why these specific actions solve it.

Then, after your analysis, generate a TEACHING that tells a student model exactly what they need to know to solve this problem.

FORMAT YOUR RESPONSE AS:

<thinking>
[Your comprehensive analysis - think freely and deeply about:
- What's really wrong here?
- Why do these specific actions fix it?
- What order must they happen in?
- What domain knowledge is needed?
- Any tricky aspects to watch for?
Take as much space as you need for thorough analysis.]
</thinking>

<teaching>
## Problem Recognition
[What symptoms in the ticket indicate this specific problem?]

## Key Insights
[What does the student MUST understand about this problem?]

## Solution Path
Step 1: {expected_actions[0]['name'] if expected_actions else 'action'}() 
   - Why: [Core reason this must happen first]
   - Watch for: [Any gotchas or important details]

Step 2: [Continue for each required action]
   - Why: [Why this follows previous step]
   - Watch for: [Important details]

[Continue for all {len(expected_actions)} actions]

## Completion Signal
CRITICAL: After completing all actions above, you MUST call done() to signal task completion.
This prevents infinite loops - the task is complete when the ticket requirements are met.

## Success Verification
The problem is solved when: [Specific criteria from ticket]
</teaching>

The <thinking> section is for YOUR analysis.
The <teaching> section is what the STUDENT needs to know to solve this problem successfully."""

    response = generate(
        model=teacher_model,
        messages=[SystemMessage(role="system", content=trace_prompt)],
        temperature=0.0  # Zero temperature for maximum consistency
    )
    
    return {
        "task_id": task.id,
        "ticket": task.ticket,
        "expected_actions": expected_actions,
        "thinking_trace": response.content,
        "trace_cost": response.cost if hasattr(response, 'cost') else 0.0
    }

def main():
    """Generate thinking traces for all telecom tasks."""
    parser = argparse.ArgumentParser(description="Generate teacher thinking traces")
    parser.add_argument("--domain", default="telecom", help="Domain to process")
    parser.add_argument("--teacher-model", default="openai/arc-teacher", 
                        help="Teacher model to use (use openai/arc-teacher for local VLLM)")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to process")
    parser.add_argument("--hard-only", action="store_true", 
                        help="Process only hard persona tasks")
    parser.add_argument("--mms-only", action="store_true",
                        help="Process only MMS issue tasks")
    parser.add_argument("--output-suffix", default="", 
                        help="Suffix for output files (e.g., '_hard' for hard-only traces)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tasks
    tasks = load_tasks(args.domain)
    
    # Get domain policy and tools
    env_constructor = registry.get_env_constructor(args.domain)
    env = env_constructor(solo_mode=True)
    domain_policy = env.get_policy()
    
    # Get tools using the proper method
    tools = env.get_tools()
    # For solo mode, also get user tools if available
    try:
        user_tools = env.get_user_tools() if hasattr(env, 'get_user_tools') else []
        all_tools = tools + user_tools
    except:
        all_tools = tools
    
    # Extract tool names
    tool_names = [tool.name for tool in all_tools]
    
    # Filter for solo-compatible tasks
    solo_tasks = [task for task in tasks if LLMSoloAgent.check_valid_task(task)]
    
    # Filter for hard tasks if requested
    if args.hard_only:
        solo_tasks = [task for task in solo_tasks if '[PERSONA:Hard]' in task.id]
        logger.info(f"Filtered to {len(solo_tasks)} hard persona tasks")
    
    # Filter for MMS tasks if requested
    if args.mms_only:
        solo_tasks = [task for task in solo_tasks if '[mms_issue]' in task.id]
        logger.info(f"Filtered to {len(solo_tasks)} MMS issue tasks")
    
    if args.limit:
        solo_tasks = solo_tasks[:args.limit]
    
    logger.info(f"Processing {len(solo_tasks)} solo-compatible tasks out of {len(tasks)} total")
    
    results = []
    total_cost = 0.0
    
    for i, task in enumerate(solo_tasks):
        logger.info(f"Processing task {i+1}/{len(solo_tasks)}: {task.id}")
        
        try:
            trace_data = generate_thinking_trace_for_task(
                task=task,
                domain_policy=domain_policy,
                tools=tool_names,
                teacher_model=args.teacher_model
            )
            results.append(trace_data)
            total_cost += trace_data["trace_cost"]
            
            # Save incrementally
            with open(OUTPUT_DIR / f"trace_{task.id}.json", "w") as f:
                json.dump(trace_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to generate trace for {task.id}: {e}")
            continue
    
    # Save complete results
    output_filename = f"all_traces{args.output_suffix}.json"
    with open(OUTPUT_DIR / output_filename, "w") as f:
        json.dump({
            "domain": args.domain,
            "teacher_model": args.teacher_model,
            "total_tasks": len(solo_tasks),
            "hard_only": args.hard_only,
            "mms_only": args.mms_only,
            "total_cost": total_cost,
            "traces": results
        }, f, indent=2)
    
    logger.info(f"Generated {len(results)} traces. Total cost: ${total_cost:.4f}")

if __name__ == "__main__":
    main()