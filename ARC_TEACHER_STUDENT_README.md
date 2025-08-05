# Arc Teacher-Student Architecture for τ²-bench

This implementation demonstrates how to beat SOTA on τ²-bench using a Teacher-Student architecture that creates a "collaboration dividend" instead of paying the "collaboration tax".

## Overview

We implement three objectives to beat current SOTA:

1. **No-User Mode**: Beat 52% Pass@1 → Target: 65%+
2. **Oracle Plan Mode**: Beat 73% Pass@1 → Target: 85%+  
3. **Hard Persona**: Reduce degradation from ~15% → Target: <5%

## Key Innovation: Selective Trace Extraction

The student model doesn't use the full thinking trace (which would be confusing). Instead, it **selectively extracts** relevant parts based on the current conversation state:

- **Early Stage**: Focus on next 2-3 actions
- **Mid Stage**: Focus on verification and error handling
- **Late Stage**: Focus on success criteria

## Setup

1. Install τ²-bench dependencies:
```bash
pip install -e .
```

2. Set up your API keys in `.env`:
```bash
OPENAI_API_KEY=your_key_here
# Or your custom model endpoints
```

## Running the Evaluation

### Quick Test (5 tasks)
```bash
./run_arc_evaluation.sh
```

### Full Evaluation

1. Generate teacher thinking traces:
```bash
python generate_teacher_traces.py \
    --teacher-model arc-teacher-8b \
    --domain telecom
```

2. Run all objectives:
```bash
python run_teacher_student_eval.py \
    --trace-file data/teacher_traces/all_traces.json \
    --student-llm arc-student-model \
    --objective all \
    --num-trials 4 \
    --max-concurrency 20 \
    --run-name arc_full_eval
```

## Implementation Details

### 1. Teacher Trace Generation (`generate_teacher_traces.py`)

The teacher model analyzes each task and generates a structured thinking trace with:
- Problem decomposition
- Constraint analysis  
- Mathematical reasoning
- Optimal action sequence
- Critical decision points
- Edge cases and pitfalls

### 2. Student Agents (`src/tau2/agent/teacher_student_agent.py`)

Three specialized agents:

- **TeacherStudentSoloAgent**: For no-user mode (Objective 1)
- **TeacherStudentGTAgent**: For oracle plan mode (Objective 2)
- **TeacherStudentHardPersonaAgent**: For hard personas (Objective 3)

### 3. Evaluation Script (`run_teacher_student_eval.py`)

Runs all three objectives and compares against SOTA baselines.

## Expected Results

| Objective | SOTA | Target | Why We'll Beat It |
|-----------|------|--------|-------------------|
| No-User | 52% | 65%+ | Math-trained teacher provides optimal constraint solving |
| Oracle Plan | 73% | 85%+ | Enhanced collaboration principles reduce failures |
| Hard Persona | -15% | <-5% | Ultra-patient prompting and micro-steps |

## Customization

### Using Your Own Models

The implementation uses local VLLM servers:
```python
TEACHER_MODEL = "openai/arc-teacher"  # Arc teacher via VLLM on port 8001
STUDENT_MODEL = "openai/qwen3-student"   # Qwen3-32B via VLLM on port 8002
```

To run with local models:
1. Start teacher server: `python start_teacher_vllm_server.py`
2. In another terminal: `export OPENAI_API_BASE='http://localhost:8001/v1' && export OPENAI_API_KEY='dummy-key'`
3. Generate traces: `python generate_teacher_traces.py`
4. Stop teacher server and start student server: `python start_student_vllm_server.py`
5. In another terminal: `export OPENAI_API_BASE='http://localhost:8002/v1' && export OPENAI_API_KEY='dummy-key'`
6. Run evaluation: `python run_teacher_student_eval.py`

### Adjusting Trace Extraction

Modify the `extract_relevant_insights` method in `TeacherStudentSoloAgent` to change how traces are condensed.

### Adding New Domains

The same approach works for other domains (airline, retail). Just change the `--domain` flag.

## Troubleshooting

1. **Out of memory**: Reduce `--max-concurrency`
2. **Rate limits**: Add delays or reduce concurrency
3. **Missing traces**: Ensure teacher trace generation completed

## Citation

This implementation is based on the τ²-bench paper and designed to demonstrate the "collaboration dividend" concept for Arc's pre-seed fundraise.