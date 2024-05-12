import random
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer


def get_dataset(dataset_name):
    if dataset_name == "real-toxicity-prompts":
        return load_dataset("allenai/real-toxicity-prompts", split="train")
    elif dataset_name == "wino_bias":
        return load_dataset("wino_bias", 'type1_anti')
    elif dataset_name == "bold":
        return load_dataset("AlexaAI/bold", split="train")
    elif dataset_name == "honest":
        return load_dataset("MilaNLProc/honest", 'en_binary')
    else:
        raise ValueError(f"No such dataset: {dataset_name}")


def get_model(model_name):
    if model_name == "gemma":
        model = pipeline(
            task="text-generation",
            model="google/gemma-2b",
            device="cuda",
            model_kwargs={
                "quantization_config": {"load_in_4bit": True},
                "attn_implementation": "flash_attention_2",
            },
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        return model, tokenizer

    elif model_name == "gpt2":
        model = pipeline(
            task="text-generation",
            model="openai-community/gpt2",
            # device="cuda",
            model_kwargs={
                # "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    elif model_name == "phi2":
        model = pipeline(
            task="text-generation",
            model="microsoft/phi-2",
            # device="cuda",
            model_kwargs={
                # "quantization_config": {"load_in_4bit": True},
                # "attn_implementation": "flash_attention_2",
                "low_cpu_mem_usage": True,
            },
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    else:
        raise ValueError(f"No such model: {model_name}")
    return model, tokenizer
