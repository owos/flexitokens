#!/usr/bin/env python3
# Compression rate evaluation script for FlexiTokens models

import os 
import torch
import transformers 
import torch
import torch.nn as nn
import numpy as np
from src.model.fxt import FxTTransformerLM, FxTAverageSingleInputWithPadding
from transformers import AutoTokenizer
import json
import torch
from src.utils.data_utils import insert_special_token
from datetime import datetime
import argparse
import evaluate
from transformers import default_data_collator
from transformers import set_seed
import functools
from datasets import DatasetDict, load_dataset
from src.eval.evaluation import evaluate_inidiv_dataset_LM
from accelerate import Accelerator, DistributedDataParallelKwargs
from src.train.train import get_model_config

os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
transformers.logging.set_verbosity_error()

accuracy_metric = evaluate.load("accuracy")
flores_language_mapping = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",
    "be": "bel_Cyrl",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}


# define args parser
def pargs():
    parser = argparse.ArgumentParser(description='Fine-tune FlexiTokens on a downstream task')
    parser.add_argument('--task', type=str, default="flores", help='The task to fine-tune on')
    parser.add_argument('--model_path', type=str, default=None, help='The path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default="model_ckpts/oscar_cyrl10x_latin5x_deva13x_baseline_1_bp", help='The output directory for the fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size for training')   
    parser.add_argument('--seed', type=int, default=42, help='The random seed')

    args = parser.parse_args()

    return args


def load_flores_dataset(target_languages, tokenizer, config=None):
    dataset_dict = DatasetDict()
    for lang in target_languages:
        # script_id = tokenizer.convert_tokens_to_ids(config['language_to_script'].get(lang))
        dataset = load_dataset("facebook/flores", flores_language_mapping[lang], split = "devtest", trust_remote_code=True)
        dataset = dataset.rename_columns({"sentence": "text"})
        dataset_dict[lang] = dataset 
    dataset_dict.keys()
    return dataset_dict

def load_model_and_tokenizer(model_config, device):
    model_config["learn_prior"] = False
    state_dict = torch.load(f"{model_config['output_dir']}/model.pth")
    model_args = get_model_config(model_config, FxTTransformerLM)
    base_model = FxTTransformerLM(**model_args)
    base_model.load_state_dict(state_dict["model"])
    
    print(base_model)

    tokenizer_path = model_config['tokenizer_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, extra_ids=0, cache_dir=model_config["cache_dir"],
        additional_special_tokens=model_config["script_tokens"])

    return base_model, tokenizer 

def tokenize_dataset(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=511)


def prepare_dataset(dataset, tokenizer, config=None):
    script_ids = [tokenizer.convert_tokens_to_ids(config['language_to_script'].get(lang_id, list(config['language_to_script'].values())[-1])) for lang_id in dataset.keys()]
    tokenized_dataset = dataset.map(functools.partial(tokenize_dataset, tokenizer=tokenizer), batched=False, remove_columns=remove_columns)
    for i, lang_id in enumerate(dataset.keys()):
        tokenized_dataset[lang_id] = tokenized_dataset[lang_id].map(functools.partial(insert_special_token, script_id=script_ids[i]))   
    return tokenized_dataset
    

def get_average_bpe_length(tokenizer, dataset):
    # Get the average length of the a sentence with a tokenizer
    for lang_id in dataset.keys():
        dataset[lang_id] = dataset[lang_id].map(tokenize_dataset, fn_kwargs={"tokenizer": tokenizer}, batched=False, remove_columns=["text"])
    results = {}
    for lang_id in dataset.keys():
        total_length = []
        for example in dataset[lang_id]:
            bpe_length = example["input_ids"].__len__()
            total_length.append(bpe_length)
    
        average_bytes = np.mean(total_length)
        results[lang_id] = average_bytes
    return results




def write_results(args,languages_bpc_dictionary):
    # Save results and model name to the file name.
    model_name = args.model_path.split("/")[-2]
    results_path = os.path.join(args.model_path, f"{model_name}_language_flores_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    
    with open(results_path, 'w') as f:
        json.dump(languages_bpc_dictionary, f, indent=4)

def main():
    args = pargs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_languages = ["en", "es", "ru", "uk", "hi", "te", "ur"]
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["project_dir"] = args.model_path
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerator_log_kwargs)
   
    # Load the safetensor file
    if args.model_path is None:
        args.model_path = "model_ckpts/oscar_cyrl10x_latin5x_deva13x_baseline_1_bp/_2024-10-28_01-12-56"
    model_config = json.load(open(f"{args.model_path}/config.json"))  
    #set num_predictors in model config:
    model_config["num_predictors"] = 3 if "3_bp" in model_config["output_dir"] else 1
    model_config["cache_dir"] = "~/owos/.cache"
    
    #dataset function mapping
  
    model, tokenizer = load_model_and_tokenizer(model_config, device)
    print("Model Loaded Successfully")
    dataset = load_flores_dataset(target_languages, tokenizer, model_config)
    if tokenizer.vocab_size > 1000: # i.e if BPE tokenizer, I don't need to enter the model, I just to tokenize the dataset and get the average len of sentence
        tokenizer.pad_token = tokenizer.eos_token
        languages_bpc_dictionary = get_average_bpe_length(tokenizer, dataset)
        print("Average BPE Length by Language:")
    else:
        tokenized_dataset = prepare_dataset(dataset, tokenizer, model_config)
        
        model = accelerator.prepare(model)

        print("Evaluating dataset...")
        
        languages_bpc_dictionary = evaluate_inidiv_dataset_LM(
            tokenized_dataset,
            default_data_collator,
            args.batch_size,
            accelerator,
            model,
        )

    print(f"Saving evaluation results.")
    if accelerator.is_main_process:
        write_results(args, languages_bpc_dictionary)
        print("Evaluation complete!")

    

if __name__ == "__main__":
    main()

# python src/eval/compression_rate.py --task flores --model_path model_ckpts/fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial/_2025-04-29_16-29-47
