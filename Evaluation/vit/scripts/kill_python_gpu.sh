#!/bin/bash
# List processes on GPU 0, 1, and 67, filter for Python processes, and kill them

# Define the GPUs
GPUS="2,4,5"

# List the processes on the GPUs and filter for Python processes
for GPU in $(echo $GPUS | tr "," "\n"); do
    echo "Killing processes on GPU $GPU..."
    # List processes using nvidia-smi, filter for Python, and extract the process IDs
    PIDS=$(nvidia-smi -i $GPU | grep python | awk '{print $5}')
    
    # Kill each process
    for PID in $PIDS; do
        echo "Killing process $PID..."
        kill -9 $PID
    done
done
