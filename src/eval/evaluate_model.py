#!/usr/bin/env python3
# Evaluation script to a pretrained FlexiTokens model
#run with:
# python src/eval/evaluate_model.py --model_path /path/to/model --output_dir /path/to/output --eval_split test
import os
os.environ["HOME"] = "~/owos/.cache/"
import argparse
import torch
import json
import logging
import math
from datetime import datetime
from collections import defaultdict
from accelerate.logging import get_logger
from transformers import DataCollatorForLanguageModeling, set_seed, AutoTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm

from src.utils import utils
from src.utils.data_utils import FxTDataset, MixtureByteVocab
from src.eval.evaluation import evaluate_inidiv_dataset_LM
from src.model.fxt import FxTTransformerLM
from src.utils.utils import init_seed, get_model_config
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained FlexiTokens model')
    
    parser.add_argument('--model_path', type=str, default=None, required=True,
                        help='Path to the model directory containing checkpoint and config.json')
    parser.add_argument('--checkpoint_name', type=str, default='model.pth',
                        help='Name of the checkpoint file within the model directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval_split', type=str, default='test',
                        choices=['test', 'validation'],
                        help='Split to evaluate on (test or validation)')
    parser.add_argument('--tokenizer_path', type=str, default='google/byt5-small',
                        help='Path to the tokenizer')

    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration from the model's config.json
    checkpoint_path = os.path.join(args.model_path, args.checkpoint_name)
    config_path = os.path.join(args.model_path, 'config.json')
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    # Initialize accelerator for distributed evaluation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(accelerator.state, main_process_only=False) 
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Handle pretrained models
    ft_model_config = None
    if "pretrained_path" in model_config:
        ft_model_config = model_config
        model_config_path = os.path.join(model_config['pretrained_path'], 'config.json')
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
    
    # Set cache directory
    model_config["cache_dir"] = "~/owos/.cache"
    
    
    if '<eot>' not in model_config["script_tokens"]:
        model_config["script_tokens"].append('<eot>')
    
    model_config["seed"] = args.seed
    model_config["eval_batch_size"] = args.eval_batch_size
    model_config["output_dir"] = args.output_dir
    
    for key, value in model_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    init_seed(args.seed)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Initialize accelerator for distributed evaluation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(accelerator.state, main_process_only=False)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare id to script mapping
    id_to_script = {value: key for key, value in model_config["id_to_script"].items()}
    language_to_script_id = {lang: int(id_to_script[script]) for lang, script in model_config["language_to_script"].items()}
    
    logger.info(f"language_to_script_id is {language_to_script_id}")
    
    # Create byte vocabulary
    tokenizer_path = "google/byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        extra_ids=0,
        cache_dir=model_config["cache_dir"],
        add_eos_token=False,
        add_prefix_space=True,
        additional_special_tokens=model_config["script_tokens"]
    )
    
    # Set up dataset
    boundary_kwargs = {
        'boundaries_type': args.boundaries_type,
        'fixed_sf': args.fixed_sf,
        'tokenizer_path': args.tokenizer_path,
        'script_tokens': args.script_tokens,
        'cache_dir': args.cache_dir,
    }
    data_corpus = FxTDataset(
        args.data, args.seq_len, accelerator, language_to_script_id, args, **boundary_kwargs

    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
    
    # Get model configuration and initialize model
    logger.info("Loading model from checkpoint")
    m_model_config = get_model_config(model_config, FxTTransformerLM)
    model = FxTTransformerLM(**m_model_config)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    if ft_model_config is not None:
        ft_state_dict = torch.load(checkpoint_path, map_location=device)["model"]
        pretrained_state_dict = {}
        for key, value in ft_state_dict.items():
            if key.startswith('score.'):
                continue
            if key.startswith('memtransformer.'):
                new_key = key[len('memtransformer.'):]
                pretrained_state_dict[new_key] = value
        model.load_state_dict(pretrained_state_dict)
        logger.info("✅ Loaded finetuned model's weights")
    else:
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict["model"])
            logger.info("✅ Loaded pretrained model")
        except:
            try:
                from safetensors.torch import load_file
                state_dict = load_file(f"{args.model_path}/model.safetensors")
                model.load_state_dict(state_dict)
                logger.info("✅ Loaded model from .safetensors")
            except:
                raise RuntimeError(f"Failed to load model from {checkpoint_path}")
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    
    # Evaluate individual languages
    logger.info("Starting evaluation on individual languages")
    split = "test" if args.eval_split == "test" else "validation"

    languages_bpc_dictionary = evaluate_inidiv_dataset_LM(
        data_corpus.individual_test_dataset if split == "test" else data_corpus.individual_validation_dataset,
        data_collator,
        args.eval_batch_size,
        accelerator,
        model
    )
    
    # Save results and model name to the file name.
    model_name = args.model_path.split("/")[-2]
    results_path = os.path.join(args.output_dir, f"{model_name}_language_{split}_eval_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    logger.info(f"Saving evaluation results to {results_path}")
    
    # Only write results on main process
    if accelerator.is_main_process:
        with open(results_path, 'w') as f:
            json.dump(languages_bpc_dictionary, f, indent=4)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()