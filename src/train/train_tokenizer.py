from transformers import AutoTokenizer
import os
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import numpy as np
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import tokenizers.pre_tokenizers as pre_tokenizers_fast
from collections import Counter
import itertools
import shutil
from datasets import DatasetDict, IterableDatasetDict, load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers



def batch_iterator(dataset, batch_size):
    for i in tqdm(range(0, len(dataset), batch_size)):
        yield dataset[i : i + batch_size]["text"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a tokenizer.")
    parser.add_argument("--template_path", type=str, default="meta-llama/Llama-3.1-8B", help="Name of the pre-trained model to use.")
    parser.add_argument("--data_path", type=str, default="~/owos/experiments/fxt/data/fineweb/", help="Path to the training data.")
    parser.add_argument("--output_dir", type=str, default="./data/bpe_tokenizer", help="Directory to save the trained tokenizer.")
    parser.add_argument("--vocab_size", type=int, default=32200, help="Vocabulary size for the tokenizer.")
    return parser.parse_args()

def calculate_fertility_for_text(text, tokenizer):
    """Helper function to calculate the number of tokens and characters for a single text."""
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    num_characters = len(text)
    return num_tokens, num_characters

def calculate_fertility(tokenizer, texts):
    """
    Calculate the fertility of a tokenizer, i.e., the average number of tokens per character in the input texts.

    Args:
        tokenizer: The trained tokenizer.
        texts: A list of sentences or text samples.
        num_workers: Number of worker processes to use for multiprocessing.
    
    Returns:
        The average fertility of the tokenizer.
    """
    total_tokens = 0
    total_characters = 0
    num_workers = max(1, os.cpu_count() - 2)
    tokens = texts.map(
        lambda x: tokenizer(x["text"], padding=False)
        , batched=True,
        batch_size=200,
        desc="Tokenzing the data",
        num_proc=num_workers)

    with mp.Pool(processes=num_workers) as pool:
        results = []
        for text_batch in tqdm(zip(texts["input_ids"], tokens), desc="Calculating tokenizer fertility"):
            batch_results = pool.starmap(calculate_fertility_for_text, text_batch)
            results.extend(batch_results)
    
    # Aggregate results
    for num_tokens, num_characters in results:
        total_tokens += num_tokens
        total_characters += num_characters

    fertility = total_tokens / total_characters if total_characters > 0 else 0
    return fertility

def train_tokenizer(tokenizer, data, output_dir, vocab_size=32768):
    """
    Train a tokenizer using the specified model and dataset, and calculate its fertility.

    Args:
        tokenizer (AutoTokenizer): Base tokenizer to use as template
        data: Iterator of text data for training
        output_dir (str): Directory to save the trained tokenizer.
        vocab_size (int): Size of the vocabulary for the new tokenizer
    """

    new_tokenizer_backend = Tokenizer(models.BPE())
    breakpoint()
    if hasattr(tokenizer, "backend_tokenizer"):
        new_tokenizer_backend.pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    else:
        new_tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["</s>"],  # Only add the tokens you want
        min_frequency=2
    )
    
    new_tokenizer_backend.train_from_iterator(data, trainer=trainer)
    
    # Convert to Transformers tokenizer
    from transformers import PreTrainedTokenizerFast
    new_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=new_tokenizer_backend,
        eos_token="</s>"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    new_tokenizer.save_pretrained(output_dir)





def main():
    args = parse_arguments()
    data_path = args.data_path
    old_tok_path = args.template_path
    language_to_script_id = ["en", "es", "ru", "uk", "hi", "te"]
    vocab_size = args.vocab_size
    max_samples_per_language = 1640000
    train_dict, validation_dict, test_dict = DatasetDict(), DatasetDict(), DatasetDict()
    for file_path in language_to_script_id:
        train = dataset = load_from_disk(
            os.path.join(data_path, file_path, "train"),
        ).select(range(max_samples_per_language))
        validation = load_from_disk(
            os.path.join(data_path, file_path, "validation"),
        )
        train_dict[file_path] = train
        validation_dict[file_path] = validation
        

    s_dataset = concatenate_datasets([data for data in train_dict.values()]).shuffle(seed=42)
    batch_size = 5000
    train_text_batched = batch_iterator(s_dataset, batch_size)
    
    source_tokenizer = AutoTokenizer.from_pretrained(old_tok_path)

    output_dir = f"./data/bpe_tokenizer_{vocab_size}"


    train_tokenizer(source_tokenizer, train_text_batched, output_dir, vocab_size=vocab_size)

if __name__ == "__main__":
    main()
