import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
You will be given an English word and you need to reverse the order of letters in the word.
### For example: 
Input: 'httpstatus', Output: 'sutatsptth'.
Input: 'hello', Output: 'olleh'.
Input: 'world', Output: 'dlrow'.
Input: 'python', Output: 'nohtyp'.
Input: 'programming', Output: 'gnimmargorp'.
"""

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> tuple[bool, int]:
    """Run the prompt up to NUM_RUNS_TIMES and return (success, runs_attempted).

    Prints "SUCCESS" when a match is found.
    Returns:
        tuple: (True if any output matches EXPECTED_OUTPUT, number of runs attempted)
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True, idx + 1  # Return success and number of runs it took
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False, NUM_RUNS_TIMES  # Return failure and total runs attempted

if __name__ == "__main__":
    # Run test_your_prompt 20 times and record success rate
    num_trials = 20
    trial_successes = 0
    total_runs = 0
    successful_runs = 0
    
    print(f"Running {num_trials} trials...\n")
    print("=" * 60)
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        success, runs_attempted = test_your_prompt(YOUR_SYSTEM_PROMPT)
        
        # Track trial-level success
        if success:
            trial_successes += 1
        
        # Track run-level statistics
        total_runs += runs_attempted
        if success:
            # If trial succeeded, the last run was successful
            successful_runs += 1
        # Note: Failed runs are already counted in runs_attempted
        
        print("=" * 60)
    
    # Calculate success rates
    trial_success_rate = (trial_successes / num_trials) * 100
    run_success_rate = (successful_runs / total_runs) * 100
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"\nTrial-Level Statistics:")
    print(f"  Total Trials: {num_trials}")
    print(f"  Successful Trials: {trial_successes}")
    print(f"  Failed Trials: {num_trials - trial_successes}")
    print(f"  Trial Success Rate: {trial_success_rate:.1f}%")
    print(f"\nRun-Level Statistics:")
    print(f"  Total Runs: {total_runs}")
    print(f"  Successful Runs: {successful_runs}")
    print(f"  Failed Runs: {total_runs - successful_runs}")
    print(f"  RUN SUCCESS RATE: {run_success_rate:.1f}%")
    print(f"{'=' * 60}")