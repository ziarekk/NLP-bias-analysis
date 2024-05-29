import random
import torch
import numpy as np
import argparse
from utils import get_model
import json
from pipelines import toxicity_dataset_inference, wino_bias_dataset_inference, bold_dataset_inference, honest_dataset_inference


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["gpt2", "gemma", "phi2"], required=True)
    parser.add_argument('--dataset', choices=["toxicity", "wino_bias", "bold", "honest"], required=True)
    
    args = parser.parse_args()
    
    # Load model
    text_generation, tokenizer = get_model(args.model)

    # Inference
    if args.dataset == "toxicity":
        scores = toxicity_dataset_inference(text_generation)
        metric = "toxicity"
    elif args.dataset == "wino_bias":
        scores = wino_bias_dataset_inference(text_generation)
        metric = "toxicity"
    elif args.dataset == "bold":
        scores = bold_dataset_inference(text_generation)
        metric = "regard"
    elif args.dataset == "honest":
        scores = honest_dataset_inference(text_generation, tokenizer)
        metric = "honest"


    result_dict = {
        "model": args.model,
        "dataset": args.dataset,
        "metric": metric,
        "scores": scores,
    }

    path = f"results/{metric}_{args.model}_{args.dataset}.json"
    with open(path, 'w') as fp:
        json.dump(result_dict, fp, indent=4)


if __name__ == '__main__':
    main()
