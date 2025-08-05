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
    
    trace_prompt = f"""You are an expert problem solver analyzing a technical support ticket.
Your strength is mathematical reasoning and constraint satisfaction.

TICKET: {task.ticket}

POLICY CONSTRAINTS:
{domain_policy}

AVAILABLE TOOLS:
{json.dumps(tools, indent=2)}

EXPECTED ACTIONS (for validation):
{json.dumps(expected_actions, indent=2)}

Generate a comprehensive thinking trace with the following structure:

## 1. PROBLEM DECOMPOSITION
- Core Issue: [What is fundamentally broken?]
- Initial State: [Current system state based on ticket]
- Target State: [What defines success?]
- Success Criteria: [Specific measurable outcomes]

## 2. CONSTRAINT ANALYSIS
- Policy Constraints: [Which policies apply?]
- Tool Constraints: [What can/cannot be done with available tools?]
- Sequencing Constraints: [What must happen in what order?]
- User Constraints: [What limitations does the user have?]

## 3. MATHEMATICAL REASONING
Frame this as a constraint satisfaction problem:
- Variables: [What can be changed?]
- Constraints: [What rules must be satisfied?]
- Objective: [What are we optimizing?]
- Solution Space: [What are valid solutions?]

## 4. OPTIMAL ACTION SEQUENCE
For each step, provide:
```
Step N:
- Tool: [exact tool name]
- Arguments: {{exact arguments with correct types}}
- Purpose: [why this step?]
- Expected Result: [what changes?]
- Verification: [how to confirm success?]
- Failure Handling: [what if this fails?]
```

## 5. CRITICAL DECISION POINTS
Identify key moments where the agent must make decisions:
- Decision 1: [description] → Choose [option] because [reasoning]
- Decision 2: [description] → Choose [option] because [reasoning]

## 6. EDGE CASES AND PITFALLS
- Common Failure 1: [description] → Prevention: [strategy]
- Common Failure 2: [description] → Prevention: [strategy]

## 7. EXECUTION HINTS
Key insights the student model should focus on:
- CRITICAL: [Most important thing to remember]
- VERIFY: [What must be checked]
- AVOID: [Common mistakes to avoid]

Think step-by-step using mathematical logic and prove your solution is correct."""

    response = generate(
        model=teacher_model,
        messages=[SystemMessage(role="system", content=trace_prompt)],
        temperature=0.1  # Low temperature for consistency
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
    parser.add_argument("--teacher-model", default="gpt-4-1106-preview", 
                        help="Teacher model to use")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to process")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tasks
    tasks = load_tasks(args.domain)
    
    # Get domain policy and tools
    env_constructor = registry.get_env_constructor(args.domain)
    env = env_constructor(solo_mode=True)
    domain_policy = env.get_info().policy
    tools = [tool.name for tool in env.tools]
    
    # Filter for solo-compatible tasks
    solo_tasks = [task for task in tasks if LLMSoloAgent.check_valid_task(task)]
    
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
                tools=tools,
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
    with open(OUTPUT_DIR / "all_traces.json", "w") as f:
        json.dump({
            "domain": args.domain,
            "teacher_model": args.teacher_model,
            "total_tasks": len(solo_tasks),
            "total_cost": total_cost,
            "traces": results
        }, f, indent=2)
    
    logger.info(f"Generated {len(results)} traces. Total cost: ${total_cost:.4f}")

if __name__ == "__main__":
    main()