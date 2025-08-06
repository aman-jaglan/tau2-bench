#!/usr/bin/env python3
"""
Run Teacher-Student evaluation on τ²-bench.
Three objectives to beat SOTA:
1. No-User mode: Beat 52% Pass@1
2. Oracle Plan mode: Beat 73% Pass@1  
3. Hard Persona: <5% degradation
"""
import json
from pathlib import Path
import argparse
from typing import Dict, List, Any
import sys

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tau2.run import run_tasks, get_tasks
from tau2.data_model.simulation import RunConfig, Results
from tau2.registry import registry
from tau2.evaluator.evaluator import EvaluationType
from tau2.agent.llm_agent import LLMSoloAgent
from tau2.metrics.agent_metrics import compute_metrics
from tau2.utils.display import ConsoleDisplay

# Import our custom agents
from tau2.agent.teacher_student_agent import (
    TeacherStudentSoloAgent, 
    TeacherStudentGTAgent,
    TeacherStudentHardPersonaAgent
)

def load_thinking_traces(trace_file: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load pre-generated thinking traces and convert to structured execution plans."""
    import re
    
    with open(trace_file) as f:
        data = json.load(f)
    
    # Create mapping from task_id to execution plan
    traces = {}
    for trace_data in data["traces"]:
        full_trace = trace_data["thinking_trace"]
        # Extract only the teaching tag content
        if "<teaching>" in full_trace and "</teaching>" in full_trace:
            start_idx = full_trace.find("<teaching>") + len("<teaching>")
            end_idx = full_trace.find("</teaching>")
            teaching_content = full_trace[start_idx:end_idx].strip()
            
            # Pre-process teaching content to extract execution plan
            execution_plan = preprocess_teaching_to_execution_plan(teaching_content)
            traces[trace_data["task_id"]] = execution_plan
            logger.debug(f"Extracted execution plan for {trace_data['task_id']} ({len(execution_plan)} steps)")
        else:
            # No teaching tag - let model work without teaching
            logger.warning(f"No teaching tag found for {trace_data['task_id']}, skipping trace")
            # Don't add to traces dict - model will work without teaching
    
    logger.info(f"Loaded {len(traces)} execution plans from {trace_file}")
    return traces


def preprocess_teaching_to_execution_plan(teaching_content: str) -> List[Dict[str, Any]]:
    """Convert teaching content to a structured execution plan."""
    import re
    
    # Find all Step N: entries
    step_pattern = r'Step \d+: ([a-zA-Z_]+)\((.*?)\)'
    steps = re.findall(step_pattern, teaching_content)
    
    tool_calls = []
    for func_name, args_str in steps:
        # Parse arguments
        args_str = args_str.strip()
        if not args_str:
            # No arguments
            args = {}
        elif args_str.startswith('{'):
            # JSON-style arguments
            try:
                args = json.loads(args_str)
            except:
                # If JSON parsing fails, try to extract key-value pairs
                args = {}
                logger.warning(f"Failed to parse JSON args: {args_str}")
        else:
            # String argument (e.g., "4g_5g_preferred")
            # Assume it's a mode parameter for network preference
            if func_name == "set_network_mode_preference":
                args = {"mode": args_str.strip('"')}
            else:
                args = {"value": args_str.strip('"')}
        
        tool_call = {
            "name": func_name,
            "arguments": args
        }
        tool_calls.append(tool_call)
    
    # Check if done() is mentioned in completion signal
    if "done()" in teaching_content.lower() or "completion signal" in teaching_content.lower():
        tool_calls.append({
            "name": "done",
            "arguments": {}
        })
    
    return tool_calls

def register_custom_agents(execution_plans: Dict[str, List[Dict[str, Any]]]):
    """Register our teacher-student agents."""
    
    # Store execution plans in a way accessible to agent factories
    global _execution_plans
    _execution_plans = execution_plans
    
    # Create wrapper classes that include execution plans
    class TSoloAgent(TeacherStudentSoloAgent):
        def __init__(self, tools, domain_policy, task, llm=None, llm_args=None):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                task=task,
                execution_plans=_execution_plans,
                student_llm=llm,
                student_llm_args=llm_args
            )
    
    class TGTAgent(TeacherStudentGTAgent):
        def __init__(self, tools, domain_policy, task, llm=None, llm_args=None):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                task=task,
                student_llm=llm,
                student_llm_args=llm_args
            )
    
    class THardAgent(TeacherStudentHardPersonaAgent):
        def __init__(self, tools, domain_policy, task, llm=None, llm_args=None):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                task=task,
                student_llm=llm,
                student_llm_args=llm_args
            )
    
    # Register agents
    registry.register_agent(TSoloAgent, "teacher_student_solo")
    registry.register_agent(TGTAgent, "teacher_student_gt")
    registry.register_agent(THardAgent, "teacher_student_hard")
    
    logger.info("Registered custom teacher-student agents")

def run_objective_1(args):
    """Objective 1: Beat No-User mode SOTA (52%)."""
    print("\n" + "="*60)
    print("OBJECTIVE 1: No-User Mode")
    print("Target: Beat 52% Pass@1")
    print("="*60 + "\n")
    
    # Get tasks
    tasks = get_tasks("telecom", task_ids=None, num_tasks=args.num_tasks)
    
    # Filter for solo-compatible tasks
    solo_tasks = [task for task in tasks if LLMSoloAgent.check_valid_task(task)]
    
    # Further filter to only MMS tasks since we only have traces for those
    solo_tasks = [task for task in solo_tasks if "mms_issue" in task.id]
    logger.info(f"Running {len(solo_tasks)} MMS solo-compatible tasks")
    
    # Run evaluation with debug logging
    logger.info(f"Starting evaluation with {len(solo_tasks)} tasks")
    
    # Set debug level for more detailed logs
    import logging
    logging.getLogger("tau2.agent.teacher_student_agent").setLevel(logging.DEBUG)
    logging.getLogger("tau2.utils.llm_utils").setLevel(logging.DEBUG)
    
    results = run_tasks(
        domain="telecom",
        tasks=solo_tasks,
        agent="teacher_student_solo",
        user="dummy_user",
        llm_agent=args.student_llm,
        llm_args_agent={"temperature": 0.0},
        llm_user="gpt-4.1-2025-04-14",  # Dummy user doesn't use this
        llm_args_user={"temperature": 0.0},
        num_trials=args.num_trials,
        max_steps=100,
        save_to=Path(f"data/simulations/teacher_student_obj1_{args.run_name}.json"),
        evaluation_type=EvaluationType.ALL,
        max_concurrency=args.max_concurrency,
        console_display=True
    )
    
    # Calculate and display metrics
    metrics = compute_metrics(results)
    print(f"\nObjective 1 Results:")
    print(f"Pass@1: {metrics.pass_hat_ks.get(1, 0)*100:.1f}% (Target: >52%)")
    print(f"Average Reward: {metrics.avg_reward:.3f}")
    
    return results, metrics

def run_objective_2(args):
    """Objective 2: Beat Oracle Plan mode SOTA (73%)."""
    print("\n" + "="*60)
    print("OBJECTIVE 2: Oracle Plan Mode")
    print("Target: Beat 73% Pass@1")
    print("="*60 + "\n")
    
    # Get tasks
    from tau2.agent.llm_agent import LLMGTAgent
    tasks = get_tasks("telecom", task_ids=None, num_tasks=args.num_tasks)
    
    # Filter for GT-compatible tasks
    gt_tasks = [task for task in tasks if LLMGTAgent.check_valid_task(task)]
    logger.info(f"Running {len(gt_tasks)} GT-compatible tasks")
    
    # Run evaluation
    results = run_tasks(
        domain="telecom",
        tasks=gt_tasks,
        agent="teacher_student_gt",
        user="user_simulator",
        llm_agent=args.student_llm,
        llm_args_agent={"temperature": 0.0},
        llm_user="gpt-4.1-2025-04-14",
        llm_args_user={"temperature": 0.7},
        num_trials=args.num_trials,
        max_steps=100,
        save_to=Path(f"data/simulations/teacher_student_obj2_{args.run_name}.json"),
        evaluation_type=EvaluationType.ALL,
        max_concurrency=args.max_concurrency,
        console_display=True
    )
    
    # Calculate and display metrics
    metrics = compute_metrics(results)
    print(f"\nObjective 2 Results:")
    print(f"Pass@1: {metrics.pass_hat_ks.get(1, 0)*100:.1f}% (Target: >73%)")
    print(f"Average Reward: {metrics.avg_reward:.3f}")
    
    return results, metrics

def run_objective_3(args):
    """Objective 3: Test robustness on Hard Persona."""
    print("\n" + "="*60)
    print("OBJECTIVE 3: Hard Persona Robustness")
    print("Target: <5% degradation from baseline")
    print("="*60 + "\n")
    
    # Get all tasks
    from tau2.agent.llm_agent import LLMGTAgent
    all_tasks = get_tasks("telecom", task_ids=None, num_tasks=None)
    
    # Filter GT-compatible tasks
    gt_tasks = [task for task in all_tasks if LLMGTAgent.check_valid_task(task)]
    
    # Separate normal and hard persona tasks
    normal_tasks = [t for t in gt_tasks if "[PERSONA:Hard]" not in t.id]
    hard_tasks = [t for t in gt_tasks if "[PERSONA:Hard]" in t.id]
    
    # Limit if requested
    if args.num_tasks:
        normal_tasks = normal_tasks[:args.num_tasks]
        hard_tasks = hard_tasks[:args.num_tasks]
    
    logger.info(f"Found {len(normal_tasks)} normal tasks and {len(hard_tasks)} hard persona tasks")
    
    # Run on normal tasks first (baseline)
    print("\nRunning baseline (normal tasks)...")
    normal_results = run_tasks(
        domain="telecom",
        tasks=normal_tasks,
        agent="teacher_student_gt",
        user="user_simulator",
        llm_agent=args.student_llm,
        llm_args_agent={"temperature": 0.0},
        llm_user="gpt-4.1-2025-04-14",
        llm_args_user={"temperature": 0.7},
        num_trials=args.num_trials,
        max_steps=100,
        save_to=Path(f"data/simulations/teacher_student_obj3_normal_{args.run_name}.json"),
        evaluation_type=EvaluationType.ALL,
        max_concurrency=args.max_concurrency,
        console_display=True
    )
    
    normal_metrics = compute_metrics(normal_results)
    
    # Run on hard persona tasks
    print("\nRunning hard persona tasks...")
    hard_results = run_tasks(
        domain="telecom",
        tasks=hard_tasks,
        agent="teacher_student_hard",  # Use special hard persona agent
        user="user_simulator",
        llm_agent=args.student_llm,
        llm_args_agent={"temperature": 0.0},
        llm_user="gpt-4.1-2025-04-14",
        llm_args_user={"temperature": 0.7},
        num_trials=args.num_trials,
        max_steps=100,
        save_to=Path(f"data/simulations/teacher_student_obj3_hard_{args.run_name}.json"),
        evaluation_type=EvaluationType.ALL,
        max_concurrency=args.max_concurrency,
        console_display=True
    )
    
    hard_metrics = compute_metrics(hard_results)
    
    # Calculate degradation
    normal_pass1 = normal_metrics.pass_hat_ks.get(1, 0) * 100
    hard_pass1 = hard_metrics.pass_hat_ks.get(1, 0) * 100
    degradation = normal_pass1 - hard_pass1
    
    print(f"\nObjective 3 Results:")
    print(f"Normal Persona Pass@1: {normal_pass1:.1f}%")
    print(f"Hard Persona Pass@1: {hard_pass1:.1f}%")
    print(f"Degradation: {degradation:.1f}% (Target: <5%)")
    
    return {
        "normal": (normal_results, normal_metrics),
        "hard": (hard_results, hard_metrics),
        "degradation": degradation
    }

def main():
    parser = argparse.ArgumentParser(description="Run Teacher-Student evaluation on τ²-bench")
    parser.add_argument("--trace-file", type=Path, default="teacher_traces/all_traces_mms_teaching.json",
                        help="Path to teacher thinking traces")
    parser.add_argument("--student-llm", default="openai/qwen3-student",
                        help="Student model to use (use openai/qwen3-student for local VLLM)")
    parser.add_argument("--objective", choices=["1", "2", "3", "all"], 
                        default="all", help="Which objective to run")
    parser.add_argument("--num-trials", type=int, default=4,
                        help="Number of trials per task")
    parser.add_argument("--num-tasks", type=int, 
                        help="Limit number of tasks (for testing)")
    parser.add_argument("--max-concurrency", type=int, default=10,
                        help="Max concurrent simulations")
    parser.add_argument("--run-name", default="test",
                        help="Name for this run (used in output files)")
    
    args = parser.parse_args()
    
    # Load thinking traces
    if not args.trace_file.exists():
        logger.error(f"Trace file not found: {args.trace_file}")
        logger.info("Please run generate_teacher_traces.py first")
        return
    
    execution_plans = load_thinking_traces(args.trace_file)
    
    # Register custom agents
    register_custom_agents(execution_plans)
    
    # Track results
    all_results = {}
    
    # Run objectives
    if args.objective in ["1", "all"]:
        results, metrics = run_objective_1(args)
        all_results["objective_1"] = {
            "metrics": metrics.as_dict(),
            "pass_at_1": metrics.pass_hat_ks.get(1, 0) * 100
        }
    
    if args.objective in ["2", "all"]:
        results, metrics = run_objective_2(args)
        all_results["objective_2"] = {
            "metrics": metrics.as_dict(),
            "pass_at_1": metrics.pass_hat_ks.get(1, 0) * 100
        }
        
    if args.objective in ["3", "all"]:
        obj3_results = run_objective_3(args)
        all_results["objective_3"] = {
            "normal_pass_at_1": obj3_results["normal"][1].pass_hat_ks.get(1, 0) * 100,
            "hard_pass_at_1": obj3_results["hard"][1].pass_hat_ks.get(1, 0) * 100,
            "degradation": obj3_results["degradation"]
        }
    
    # Save summary
    summary_path = Path(f"data/simulations/teacher_student_summary_{args.run_name}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "student_model": args.student_llm,
            "num_trials": args.num_trials,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {summary_path}")

if __name__ == "__main__":
    main()