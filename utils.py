import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig


def get_model(model_name):
    api_key = open('keys/api_key.txt', 'r').read()
    batch_size = 64
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
	    bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if model_name == "gemma":
        model = pipeline(
            task="text-generation",
            model="google/gemma-2b",
            model_kwargs={
                "quantization_config": bnb_config,
                "attn_implementation": "flash_attention_2",
		        "low_cpu_mem_usage": True,
            },
            batch_size=batch_size,
            token=api_key,
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        return model, tokenizer

    elif model_name == "gpt2":
        model = pipeline(
            task="text-generation",
            model="openai-community/gpt2",
            model_kwargs={
                "quantization_config": bnb_config,
		        "low_cpu_mem_usage": True,
            },
            batch_size=batch_size,
            token=api_key,
        )
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    elif model_name == "phi2":
        model = pipeline(
            task="text-generation",
            model="microsoft/phi-2",
            model_kwargs={
                "quantization_config": bnb_config,
                "attn_implementation": "flash_attention_2",
		        "low_cpu_mem_usage": True,
            },
            batch_size=batch_size,
            token=api_key,
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    else:
        raise ValueError(f"No such model: {model_name}")
    return model, tokenizer
