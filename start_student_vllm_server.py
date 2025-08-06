#!/usr/bin/env python3
"""
Start VLLM server for Qwen/Qwen3-32B model.
This creates an OpenAI-compatible API endpoint that LiteLLM can use.
"""
import subprocess
import sys
import os
import time

def start_student_server():
    """Start VLLM server for the student model."""
    
    print("Starting VLLM server for Qwen/Qwen3-8B...")
    print("Using 2xH100 GPUs with 32k context window")
    
    # VLLM command with proper settings for H100s and Qwen3
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3-8B",
        "--port", "8002",  # Student on port 8002
        "--max-model-len", "32768",  # 32k context
        "--gpu-memory-utilization", "0.95",  # Use most of GPU memory
        "--tensor-parallel-size", "2",  # Use both H100s
        "--dtype", "auto",  # Let VLLM choose optimal dtype
        "--trust-remote-code",  # Qwen models often need custom code
        "--download-dir", os.path.expanduser("~/.cache/huggingface"),
        "--served-model-name", "qwen3-student"  # Alias for the model
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("\nServer will be available at: http://localhost:8002/v1")
    print("Model name for LiteLLM: openai/qwen3-student")
    print("\nTo use with tau2-bench, set:")
    print("export OPENAI_API_BASE='http://localhost:8002/v1'")
    print("export OPENAI_API_KEY='dummy-key'  # VLLM doesn't need a real key")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # Start the server
        process = subprocess.Popen(cmd)
        
        # Wait a bit for server to start
        print("\nWaiting for server to initialize...")
        time.sleep(10)
        
        print("\nServer should be running. You can test it with:")
        print("curl http://localhost:8002/v1/models")
        
        # Keep the process running
        process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        process.terminate()
        process.wait()
        print("Server stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if vllm is installed
    try:
        import vllm
        print(f"VLLM version: {vllm.__version__}")
    except ImportError:
        print("ERROR: VLLM not installed. Please install it with:")
        print("pip install vllm")
        sys.exit(1)
    
    start_student_server()