import random
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from tqdm import tqdm
import argparse
from utils import get_model
import json

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def toxicity_dataset_inference(text_generation, metric):
    toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")

    model_continuations = []
    for i, prompt in tqdm(enumerate(toxicity_prompts)):
        sample = prompt['prompt']['text']
        generation = text_generation(sample, do_sample=False, pad_token_id=50256)
        continuation = generation[0]['generated_text'].replace(sample, '')
        model_continuations.append(continuation)
        if i == 10:
            break

    toxicity_values = metric.compute(predictions=model_continuations)

    scores = {
        "min": round(min(toxicity_values["toxicity"]), 5),
        "max": round(max(toxicity_values["toxicity"]), 5),
        "avg": round(sum(toxicity_values["toxicity"]) / len(toxicity_values["toxicity"]), 5),
    }
    return scores


def wino_bias_dataset_inference(text_generation, metric):
    wino_bias = load_dataset("wino_bias", 'type1_anti')["test"]

    female_continuations = []
    male_continuations = []
    for i in tqdm(range(len(wino_bias))):
        s = " ".join(wino_bias[i]["tokens"])
        if (i - 1) % 2 == 0:
            prompt = s.split(" she ")[0] + " she"
            generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
            continuation = generation[0]['generated_text'].replace(prompt, '')
            female_continuations.append(continuation)
        else:
            prompt = s.split(" he ")[0] + " he"
            generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
            continuation = generation[0]['generated_text'].replace(prompt, '')
            male_continuations.append(continuation)
        if i == 5:
            break

    female_values = metric.compute(predictions=female_continuations)
    male_values = metric.compute(predictions=male_continuations)

    scores = {
        "f_min": round(min(female_values["toxicity"]), 5),
        "f_max": round(max(female_values["toxicity"]), 5),
        "f_avg": round(sum(female_values["toxicity"]) / len(female_values["toxicity"]), 5),
        "m_min": round(min(male_values["toxicity"]), 5),
        "m_max": round(max(male_values["toxicity"]), 5),
        "m_avg": round(sum(male_values["toxicity"]) / len(male_values["toxicity"]), 5),
    }
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["gpt2", "gemma", "phi2"], required=True)
    parser.add_argument('--dataset', choices=["toxicity", "wino_bias"], required=True)
    args = parser.parse_args()

    # Load model
    text_generation, tokenizer = get_model(args.model)
    toxicity = evaluate.load("toxicity")

    # Inference
    if args.dataset == "toxicity":
        scores = toxicity_dataset_inference(text_generation, toxicity)
    elif args.dataset == "wino_bias":
        scores = wino_bias_dataset_inference(text_generation, toxicity)

    result_dict = {
        "model": args.model,
        "dataset": args.dataset,
        "metric": "toxicity",
        "scores": scores,
    }

    path = f"results/toxicity_{args.model}_{args.dataset}.json"
    with open(path, 'w') as fp:
        json.dump(result_dict, fp, indent=4)


if __name__ == '__main__':
    main()
