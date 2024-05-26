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


def toxicity_dataset_inference(text_generation):
    toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")
    toxicity_prompts = toxicity_prompts.map(lambda x: {"prompt": x["prompt"]["text"]})
    metric = evaluate.load('toxicity', device='cuda')

    toxicity_min = 1.
    toxicity_max = 0.
    toxicity_sum = 0.
    num_samples = 19968  # len(toxicity_prompts)

    for i in tqdm(range(0, num_samples, text_generation._batch_size)):
        batch_prompts = toxicity_prompts['prompt'][i:i+text_generation._batch_size]
        outputs = text_generation(batch_prompts)

        outputs = [output[0]["generated_text"] for output in outputs]
    
        toxicity_values = metric.compute(predictions=outputs)

        toxicity_min = min(min(toxicity_values["toxicity"]), toxicity_min)
        toxicity_max = max(max(toxicity_values["toxicity"]), toxicity_max)
        toxicity_sum += sum(toxicity_values["toxicity"])

    scores = {
        "min": round(toxicity_min, 5),
        "max": round(toxicity_max, 5),
        "avg": round(toxicity_sum / num_samples, 5),
    }
    return scores


def wino_bias_dataset_inference(text_generation):
    wino_bias = load_dataset("wino_bias", 'type1_anti')["test"]
    metric = evaluate.load('toxicity', device='cuda')

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


def bold_dataset_inference(text_generation):
    bold_dataset = load_dataset("AlexaAI/bold", split="train")
    metric = evaluate.load('regard', device='cuda')

    male_prompts = bold_dataset.filter(lambda subset: subset['domain'] == 'gender' and subset['category'] == 'American_actors')
    female_prompts = bold_dataset.filter(lambda subset: subset['domain'] == 'gender' and subset['category'] == 'American_actresses' )

    male_prompts = [item['prompts'][0] for item in male_prompts]
    female_prompts = [item['prompts'][0] for item in female_prompts]

    female_num_samples = len(female_prompts)
    male_num_samples = len(male_prompts)

    female_sum_positive = 0.
    female_sum_other = 0.
    female_sum_neutral = 0.
    female_sum_negative = 0.

    male_sum_positive = 0.
    male_sum_other = 0.
    male_sum_neutral = 0.
    male_sum_negative = 0.

    num_iterations_f = 0
    num_iterations_m = 0

    for i in tqdm(range(0, female_num_samples, text_generation._batch_size)):
        batch_prompts = female_prompts[i:min(i+text_generation._batch_size, female_num_samples)]
        outputs = text_generation(batch_prompts)

        outputs = [output[0]["generated_text"] for output in outputs]
        scores = metric.compute(data=outputs, aggregation='average')

        female_sum_positive += scores['average_regard']['positive']
        female_sum_other += scores['average_regard']['other']
        female_sum_neutral += scores['average_regard']['neutral']
        female_sum_negative += scores['average_regard']['negative']
        
        num_iterations_f += 1
       
    for i in tqdm(range(0, male_num_samples, text_generation._batch_size)):
        batch_prompts = male_prompts[i:min(i+text_generation._batch_size, male_num_samples)]
        outputs = text_generation(batch_prompts)

        outputs = [output[0]["generated_text"] for output in outputs]
        scores = metric.compute(data=outputs, aggregation='average')
        
        male_sum_positive += scores['average_regard']['positive']
        male_sum_other += scores['average_regard']['other']
        male_sum_neutral += scores['average_regard']['neutral']
        male_sum_negative += scores['average_regard']['negative']
    
        num_iterations_m += 1

    final_scores = {
        "male_average_scores":{
            "positive": male_sum_positive / num_iterations_m,
            "other": male_sum_other / num_iterations_m,
            "neutral": male_sum_neutral / num_iterations_m,
            "negative": male_sum_negative / num_iterations_m 
        },
        "female_average_scores":{
            "positive": female_sum_positive / num_iterations_f,
            "other": female_sum_other / num_iterations_f,
            "neutral": female_sum_neutral / num_iterations_f,
            "negative": female_sum_negative / num_iterations_f 
        }
    }
    
    return final_scores
    

def honest_dataset_inference(text_generation, tokenizer):
    honest_dataset = load_dataset("MilaNLProc/honest", 'en_binary', split='honest')
    metric = evaluate.load('honest', 'en')
    
    # on the entier dataset
    prompts = [(h['template_masked'].replace(' [M].',''), h['category']) for h in honest_dataset]
    
    # on a sampled subeset with equal ratio of male and female prompts
    # female_prompts = random.sample([(h['template_masked'].replace(' [M].',''), h['category']) for h in honest_dataset if h['category'].startswith('female')], 40)
    # male_prompts = random.sample([(h['template_masked'].replace(' [M].',''), h['category']) for h in honest_dataset if h['category'].startswith('male')], 40)
    # prompts = female_prompts + male_prompts
   
    #num of generations
    k = 1
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
        
        # if i == 79:
        #     break

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
