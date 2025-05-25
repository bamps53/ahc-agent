#!/bin/bash
# Script to validate AHCAgent CLI on sample problem

# Set up test environment
echo "Setting up test environment..."
mkdir -p ~/ahc_test_workspace

# Install the package in development mode
echo "Installing AHCAgent CLI in development mode..."
cd /home/ubuntu/ahc_agent
pip install -e .

# Initialize AHC project
echo "Initializing AHC project..."
ahc-agent init --workspace ~/ahc_test_workspace

# Check Docker status
echo "Checking Docker status..."
ahc-agent docker status

# Solve sample problem
echo "Solving sample problem..."
ahc-agent solve /home/ubuntu/ahc_agent/sample_problem.md --workspace ~/ahc_test_workspace --time-limit 300

# List sessions
echo "Listing sessions..."
ahc-agent status

# Get session ID from the first session
SESSION_ID=$(ahc-agent status | grep "Session ID:" | head -n 1 | awk '{print $3}')

if [ -n "$SESSION_ID" ]; then
    echo "Found session: $SESSION_ID"

    # Check session status
    echo "Checking session status..."
    ahc-agent status $SESSION_ID

    # Submit best solution
    echo "Submitting best solution..."
    ahc-agent submit $SESSION_ID --output ~/ahc_test_workspace/best_solution.cpp

    echo "Best solution saved to ~/ahc_test_workspace/best_solution.cpp"
else
    echo "No session found"
fi

echo "Validation complete"
