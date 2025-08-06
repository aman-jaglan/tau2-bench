#!/usr/bin/env python3
"""
Test script to validate the preprocessing logic for teaching traces.
"""
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger

def preprocess_teaching_to_json(teaching_content: str) -> str:
    """Convert teaching content with function calls to structured JSON format."""
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
    
    # Create structured output
    structured_output = "Execute these tool calls in order:\n"
    for i, tool_call in enumerate(tool_calls):
        structured_output += f"{i+1}. {json.dumps(tool_call)}\n"
    
    structured_output += "\nIMPORTANT: Execute exactly ONE tool call per message. Wait for the tool response before proceeding to the next tool call."
    
    return structured_output


def test_preprocessing():
    """Test the preprocessing on all teaching traces."""
    trace_file = Path("teacher_traces/all_traces_mms_teaching.json")
    
    if not trace_file.exists():
        logger.error(f"Trace file not found: {trace_file}")
        return
    
    # Load traces
    with open(trace_file) as f:
        data = json.load(f)
    
    # Process each trace
    results = []
    for trace_data in data["traces"]:
        task_id = trace_data["task_id"]
        full_trace = trace_data["thinking_trace"]
        
        # Extract teaching content
        if "<teaching>" in full_trace and "</teaching>" in full_trace:
            start_idx = full_trace.find("<teaching>") + len("<teaching>")
            end_idx = full_trace.find("</teaching>")
            teaching_content = full_trace[start_idx:end_idx].strip()
            
            # Preprocess
            processed = preprocess_teaching_to_json(teaching_content)
            
            # Extract the first 500 chars of original for comparison
            original_preview = teaching_content[:500] + "..." if len(teaching_content) > 500 else teaching_content
            
            results.append({
                "task_id": task_id,
                "original_teaching": original_preview,
                "full_original": teaching_content,
                "processed_output": processed,
                "length_original": len(teaching_content),
                "length_processed": len(processed)
            })
            
            logger.info(f"Processed {task_id}")
        else:
            logger.warning(f"No teaching tag found for {task_id}")
    
    # Save results
    output_file = Path("preprocessing_test_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "total_tasks": len(results),
            "results": results
        }, f, indent=2)
    
    logger.info(f"Saved preprocessing results to {output_file}")
    
    # Print summary
    print(f"\nProcessed {len(results)} tasks")
    print(f"Results saved to: {output_file}")
    
    # Show a few examples
    print("\n=== First 3 Examples ===")
    for i, result in enumerate(results[:3]):
        print(f"\nTask {i+1}: {result['task_id']}")
        print(f"Original length: {result['length_original']} chars")
        print(f"Processed length: {result['length_processed']} chars")
        print(f"\nProcessed output:\n{result['processed_output']}")
        print("-" * 80)


if __name__ == "__main__":
    test_preprocessing()