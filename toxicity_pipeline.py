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


def bold_dataset_inference(text_generation, metric):
    bold_dataset = load_dataset("AlexaAI/bold", split="train")
    
    # on the entier dataset
    filtered_prompts = bold_dataset.filter(lambda subset: subset['domain'] == 'gender')
    
    # on a sampled subeset with equal ratio of male and female prompts
    # female_bold = (random.sample([p for p in bold_dataset if p['category'] == 'American_actresses'],4))
    # male_bold = (random.sample([p for p in bold_dataset if p['category'] == 'American_actors'],4))
    # filtered_prompts = female_bold + male_bold
    
    female_continuations = []
    male_continuations = []

    for i, prompt in tqdm(enumerate(filtered_prompts)):
        sample = prompt['prompts'][0]
        generation = text_generation(sample, max_length=50, do_sample=False, pad_token_id=50256)
        continuation = generation[0]['generated_text'].replace(sample,'')
        
        if prompt['category'] == 'American_actresses':
            female_continuations.append(continuation)
        elif prompt['category'] == 'American_actors':
            male_continuations.append(continuation)
        
        print(f'bold, e:{i}')    
        if i == 7:
            break
        
    scores = metric.compute(data = male_continuations, references= female_continuations, aggregation = 'average')
    
    scores['male_scores'] = scores.pop('average_data_regard')
    scores['female_scores'] = scores.pop('average_references_regard')
    
    return scores
    

def honest_dataset_inference(text_generation, tokenizer, metric):
    honest_dataset = load_dataset("MilaNLProc/honest", 'en_binary', split='honest')
    
    
    # on the entier dataset
    prompts = [(h['template_masked'].replace(' [M].',''), h['category']) for h in honest_dataset]
    
    # on a sampled subeset with equal ratio of male and female prompts
    # female_prompts = random.sample([(h['template_masked'].replace(' [M].',''), h['category']) for h in honest_dataset if h['category'].startswith('female')], 40)
    # male_prompts = random.sample([(h['template_masked'].replace(' [M].',''), h['category']) for h in honest_dataset if h['category'].startswith('male')], 40)
    # prompts = female_prompts + male_prompts
   
    #num of generations
    k = 20
    male_continuations = []
    female_continuations = []
    
    for i, prompt in tqdm(enumerate(prompts)):
        sample = prompt[0]      # take prompt text - 'template_masked'
        
        generation = text_generation(sample, max_length=len(tokenizer(sample)['input_ids'])+10, num_return_sequences=k, pad_token_id=50256)
        continuation = generation[0]['generated_text'].replace(sample,'')
        
        # check 'category'
        if prompt[1].startswith('male'):    
            male_continuations.append(continuation)
        elif prompt[1].startswith('female'):
            female_continuations.append(continuation)
        
        if i == 79:
            break

    female_continuations = [continuation.split() for continuation in female_continuations]
    male_continuations = [continuation.split() for continuation in male_continuations]
    
    scores = {
        "male_avg_regard": metric.compute(predictions=male_continuations),
        "female_avg_regard": metric.compute(predictions=female_continuations)
    }

    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["gpt2", "gemma", "phi2"], required=True)
    parser.add_argument('--dataset', choices=["toxicity", "wino_bias", "bold", "honest"], required=True)
    
    args = parser.parse_args()
    
    # Load model
    text_generation, tokenizer = get_model(args.model)
    toxicity = evaluate.load('toxicity')
    regard = evaluate.load('regard', 'compare')
    honest = evaluate.load('honest', 'en')

    # Inference
    if args.dataset == "toxicity":
        scores = toxicity_dataset_inference(text_generation, toxicity)
        metric = "toxicity"
    elif args.dataset == "wino_bias":
        scores = wino_bias_dataset_inference(text_generation, toxicity)
        metric = "toxicity"
    elif args.dataset == "bold":
        scores = bold_dataset_inference(text_generation, regard)
        metric = "regard"
    elif args.dataset == "honest":
        scores = honest_dataset_inference(text_generation, tokenizer, honest)
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
