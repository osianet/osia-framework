# All endpoints use scale-to-zero: 0 replicas when idle = $0 cost.
# Cold start is ~2-5 min depending on model size.

ENDPOINTS = [
    {
        "name": "osia-dolphin-8b",
        "repository": "dphn/Dolphin3.0-Llama3.1-8B",
        "task": "text-generation",
        "framework": "pytorch",
        "accelerator": "gpu",
        "vendor": "aws",
        "region": "us-east-1",
        "instance_type": "nvidia-l4",
        "instance_size": "x1",
        "type": "protected",
        "min_replica": 0,
        "max_replica": 1,
        "scale_to_zero_timeout": 15,  # minutes (min required by HF)
        "custom_image": {
            "health_route": "/health",
            "url": "ghcr.io/huggingface/text-generation-inference:latest",
            "env": {
                "MODEL_ID": "/repository",
                "MAX_INPUT_LENGTH": "4096",
                "MAX_TOTAL_TOKENS": "8192",
                "MAX_BATCH_PREFILL_TOKENS": "4096",
            },
        },
        "description": "Dolphin 3.0 8B — uncensored general-purpose model for HUMINT desk fallback",
    },
    {
        "name": "osia-dolphin-70b",
        "repository": "dphn/dolphin-2.9.1-llama-3-70b",
        "task": "text-generation",
        "framework": "pytorch",
        "accelerator": "gpu",
        "vendor": "aws",
        "region": "us-east-1",
        "instance_type": "nvidia-a100",
        "instance_size": "x2",
        "type": "protected",
        "min_replica": 0,
        "max_replica": 1,
        "scale_to_zero_timeout": 15,  # minutes (min required by HF)
        "custom_image": {
            "health_route": "/health",
            "url": "ghcr.io/huggingface/text-generation-inference:latest",
            "env": {
                "MODEL_ID": "/repository",
                "MAX_INPUT_LENGTH": "4096",
                "MAX_TOTAL_TOKENS": "8192",
                "MAX_BATCH_PREFILL_TOKENS": "4096",
            },
        },
        "description": "Dolphin 2.9.1 70B — primary high-parameter uncensored model",
    },
]
