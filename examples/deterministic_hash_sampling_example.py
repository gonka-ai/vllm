"""
Example demonstrating deterministic hash sampling in vLLM.

This example shows how to use the new DETERMINISTIC_HASH sampling mode
which provides bit-level reproducibility across different machines without
relying on floating-point randomness.
"""

from vllm import LLM, SamplingParams

def main():
    # Initialize the LLM
    llm = LLM(model="facebook/opt-125m")  # Use a small model for demo
    
    # Example prompts
    prompts = [
        "The capital of France is",
        "Once upon a time",
    ]
    
    # Create sampling params with deterministic hash sampling
    # Set use_deterministic_hash=True and provide a seed
    sampling_params = SamplingParams(
        temperature=1.0,  # Must be > 0 for non-greedy sampling
        max_tokens=20,
        seed=42,  # Seed is used for the hash function
        use_deterministic_hash=True,  # Enable deterministic hash sampling
    )
    
    print("=" * 80)
    print("Deterministic Hash Sampling Example")
    print("=" * 80)
    print(f"\nSampling Parameters:")
    print(f"  - Temperature: {sampling_params.temperature}")
    print(f"  - Seed: {sampling_params.seed}")
    print(f"  - Use Deterministic Hash: {sampling_params.use_deterministic_hash}")
    print(f"  - Sampling Type: {sampling_params.sampling_type}")
    print()
    
    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)
    
    print("\nNote: Running this script multiple times with the same seed")
    print("will produce identical outputs (bit-level reproducibility).")
    print("\nThis is different from regular random sampling which uses")
    print("floating-point random number generators and may vary slightly")
    print("across different hardware or software configurations.")
    
    # Demonstrate reproducibility
    print("\n" + "=" * 80)
    print("Demonstrating Reproducibility")
    print("=" * 80)
    
    # Run the same generation again
    outputs2 = llm.generate(prompts, sampling_params)
    
    # Check if outputs are identical
    all_identical = True
    for out1, out2 in zip(outputs, outputs2):
        if out1.outputs[0].text != out2.outputs[0].text:
            all_identical = False
            break
    
    if all_identical:
        print("\n✓ SUCCESS: Both runs produced identical outputs!")
    else:
        print("\n✗ WARNING: Outputs differ (this should not happen)")
    
    # Compare with regular random sampling
    print("\n" + "=" * 80)
    print("Comparison with Regular Random Sampling")
    print("=" * 80)
    
    regular_sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=20,
        seed=42,  # Same seed
        use_deterministic_hash=False,  # Use regular random sampling
    )
    
    print(f"\nRegular Sampling Type: {regular_sampling_params.sampling_type}")
    outputs_regular = llm.generate(prompts[:1], regular_sampling_params)
    
    print(f"\nRegular Random Sampling Output:")
    print(f"Generated: {outputs_regular[0].outputs[0].text}")
    print("\nNote: Regular random sampling may produce different results")
    print("even with the same seed on different hardware/configurations.")


if __name__ == "__main__":
    main()
