#!/bin/bash

# Run Arc Teacher-Student Evaluation on τ²-bench
# This script runs all three objectives to beat SOTA

echo "Arc Teacher-Student Evaluation Pipeline"
echo "======================================"

# Configuration
TEACHER_MODEL="Arc-Intelligence/arc-teacher-8b"
STUDENT_MODEL="Qwen/Qwen3-32B"
NUM_TRIALS=4
MAX_CONCURRENCY=10

# Step 1: Generate teacher thinking traces (if not already done)
TRACE_FILE="data/teacher_traces/all_traces.json"

if [ ! -f "$TRACE_FILE" ]; then
    echo "Step 1: Generating teacher thinking traces..."
    python generate_teacher_traces.py \
        --teacher-model $TEACHER_MODEL \
        --domain telecom \
        --limit 10  # Remove limit for full evaluation
else
    echo "Step 1: Using existing traces at $TRACE_FILE"
fi

# Step 2: Run all three objectives
echo -e "\nStep 2: Running Teacher-Student evaluation..."

# For testing, limit to 5 tasks
python run_teacher_student_eval.py \
    --trace-file $TRACE_FILE \
    --student-llm $STUDENT_MODEL \
    --objective all \
    --num-trials $NUM_TRIALS \
    --num-tasks 5 \
    --max-concurrency $MAX_CONCURRENCY \
    --run-name "arc_test_$(date +%Y%m%d_%H%M%S)"

# For full evaluation, remove --num-tasks flag:
# python run_teacher_student_eval.py \
#     --trace-file $TRACE_FILE \
#     --student-llm $STUDENT_MODEL \
#     --objective all \
#     --num-trials $NUM_TRIALS \
#     --max-concurrency $MAX_CONCURRENCY \
#     --run-name "arc_full_$(date +%Y%m%d_%H%M%S)"

echo -e "\nEvaluation complete!"