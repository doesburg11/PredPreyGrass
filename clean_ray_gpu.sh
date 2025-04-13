#!/bin/bash

echo "Shutting down Ray..."
ray stop --force

echo "Killing zombie Python processes related to Ray or training..."
pkill -9 -f "ray::"
pkill -9 -f "train_rllib"
pkill -9 -f "python"  # Be careful if you have unrelated python apps running

echo "Freeing up shared memory and file handles (Linux)..."
rm -rf /dev/shm/plasma*

echo "Checking GPU state..."
nvidia-smi

echo "Ray and Python processes cleaned up. GPU should be idle."
