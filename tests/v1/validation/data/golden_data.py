# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Golden test data for validation tests.

These values are reference data for cross-implementation verification.
Any correct implementation of the deterministic sampling should produce
identical values.
"""

# Reference seed used for generating golden values
REFERENCE_SEED = "reference_seed_v1"

# Expected u64 values from Sha256CounterRNG.next_u64()
# These should match exactly in any correct implementation
REFERENCE_U64_VALUES = [
    4286832458236889005,
    12281003819428572724,
    12352776571910749143,
    12178488218135958089,
    6205195570139478562,
    16961475390381133449,
    4266954777775371921,
    13066482787726221110,
    16734088885020042614,
    3747751605064727020,
]

# Reference uniform01 values (top 53 bits / 2^53)
REFERENCE_UNIFORM01_VALUES = [
    0.9068004449621377,
    0.4617930339411858,
    0.6654016091697149,
    0.5163785527847619,
    0.6518953972990012,
    0.5252103851316428,
    0.3389057015892847,
    0.6336096627037583,
    0.4395949424780534,
    0.4820731330936308,
]

# Reference sampling results
# Seed: "reference_sampling_v1"
# Weights: [60000, 5000, 500]
REFERENCE_SAMPLING_SEED = "reference_sampling_v1"
REFERENCE_SAMPLING_WEIGHTS = [60000, 5000, 500]
REFERENCE_SAMPLING_RESULTS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Expected indices

# Example honest artifact data
# This is what an honest executor would produce
EXAMPLE_HONEST_ARTIFACT = {
    "seed": 42,
    "prompt": "What is 2+2?",
    "temperature": 0.99,
    "tokens": [
        {
            "token": "100",
            "top_tokens": ["100", "101", "102", "103", "104"],
            "logprobs": {
                "100": -0.5,
                "101": -2.0,
                "102": -3.5,
                "103": -5.0,
                "104": -6.5,
            },
            "sampling_weights": {
                "100": 56923,
                "101": 7025,
                "102": 867,
                "103": 107,
                "104": 14,
            },
        },
    ],
}

# Example pre-fill attack artifact
# The token doesn't match what would be sampled from the weights
EXAMPLE_PREFILL_ATTACK_ARTIFACT = {
    "seed": 42,
    "prompt": "What is 2+2?",
    "temperature": 0.99,
    "tokens": [
        {
            "token": "104",  # WRONG: Would actually sample "100"
            "top_tokens": ["100", "101", "102", "103", "104"],
            "logprobs": {
                "100": -0.5,
                "101": -2.0,
                "102": -3.5,
                "103": -5.0,
                "104": -6.5,
            },
            "sampling_weights": {
                "100": 56923,
                "101": 7025,
                "102": 867,
                "103": 107,
                "104": 14,
            },
        },
    ],
}

# Example tampered weights artifact
# The weights don't match what would be computed from logprobs
EXAMPLE_TAMPERED_WEIGHTS_ARTIFACT = {
    "seed": 42,
    "prompt": "What is 2+2?",
    "temperature": 0.99,
    "tokens": [
        {
            "token": "104",
            "top_tokens": ["100", "101", "102", "103", "104"],
            "logprobs": {
                "100": -0.5,
                "101": -2.0,
                "102": -3.5,
                "103": -5.0,
                "104": -6.5,
            },
            "sampling_weights": {
                "100": 0,  # TAMPERED: Should be ~56923
                "101": 0,
                "102": 0,
                "103": 0,
                "104": 65536,  # Put all weight on last token
            },
        },
    ],
}

# Expected distance values for Go parity verification
# Format: (inf_logprobs, val_logprobs, expected_distance)
DISTANCE_TEST_CASES = [
    # Identical logprobs -> baseline distance
    (
        [{"100": -0.5, "200": -1.2}],
        [{"100": -0.5, "200": -1.2}],
        0.004975124378109453,  # (0 + 1) / (100 * 2 + 1)
    ),
    # Small difference
    (
        [{"100": -0.5, "200": -1.2}] * 100,
        [{"100": -0.55, "200": -1.25}] * 100,
        None,  # Will be computed, just check it's small
    ),
]

# Fraud detection test cases
# Format: (description, should_detect_fraud, artifact_type)
FRAUD_DETECTION_CASES = [
    ("honest_same_gpu", False, "honest"),
    ("prefill_attack", True, "prefill"),
    ("tampered_weights", True, "tampered"),
    ("wrong_model", True, "wrong_model"),
]
