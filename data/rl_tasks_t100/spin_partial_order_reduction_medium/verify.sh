#!/bin/bash

# Legacy verification script for SPIN model checker
# Author: Previous developer (departed)
# Model: mutex.pml - Mutual exclusion protocol

# TODO: Need to investigate partial order reduction settings
# The state space seems large - POR might help but not sure if it's active
# Check SPIN documentation on -DNOREDUCE flag impact
# Performance baseline needed before optimization

SPIN=/usr/local/bin/spin
MODEL=mutex.pml

echo "Starting SPIN verification process..."

# Generate verifier from PROMELA model
echo "Step 1: Generating verifier..."
$SPIN -a $MODEL

# Compile the verifier
echo "Step 2: Compiling verifier..."
cc -o pan pan.c

# Run verification
echo "Step 3: Running verification..."
./pan

echo "Verification complete."