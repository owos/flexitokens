# Import required libraries
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import os

# Define constants
BYTE_MODEL = "google/byt5-small"
CACHE_DIR = "cache"
SEED = 42

def plot_length_distribution(dataset_name, language=None, output_dir="plots"):
    """
    Create and save a plot showing the token length distribution for a dataset.
    
    Args:
        dataset_name: Name of the dataset ("xnli", "paws-x", or "medical_abstracts")
        language: Language to use for datasets that require it (e.g., "ru" for XNLI)
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BYTE_MODEL, cache_dir=CACHE_DIR)
    
    # Load and process the dataset based on its type
    if dataset_name == "xnli":
        dataset = load_dataset(dataset_name, language, cache_dir=CACHE_DIR)
        def process_data(examples):
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=False)
        
        tokenized_dataset = dataset.map(process_data, batched=True, batch_size=2000)
        title_suffix = f" ({language})"
        
    elif dataset_name == "paws-x":
        dataset = load_dataset("paws-x", language, cache_dir=CACHE_DIR, trust_remote_code=True)
        def process_data(examples):
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=False)
        
        tokenized_dataset = dataset.map(process_data, batched=True, batch_size=2000)
        title_suffix = f" ({language})"
        
    elif dataset_name == "medical_abstracts":
        dataset = load_dataset("TimSchopf/medical_abstracts", cache_dir=CACHE_DIR)
        dataset = dataset.rename_columns({"medical_abstract": "text", "condition_label": "label"})
        def process_data(examples):
            return tokenizer(examples["text"], truncation=False)
        
        tokenized_dataset = dataset.map(process_data, batched=True, batch_size=2000)
        title_suffix = ""
    
    # Get lengths of all examples from the first split (usually 'train')
    split = list(tokenized_dataset.keys())[0]  # Get the first split (train, validation, etc.)
    lengths = [len(x) for x in tokenized_dataset[split]["input_ids"]]
    
    # Calculate statistics
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    max_length = max(lengths)
    min_length = min(lengths)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(lengths, bins=50, alpha=0.7, color='skyblue', density=True)
    
    # Add a kernel density estimate
    kde_x = np.linspace(min(lengths), max(lengths), 1000)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(lengths)
    plt.plot(kde_x, kde(kde_x), color='#CC0000', linewidth=2)  # Sharp red for the density curve
    
    # Add vertical lines for mean and median
    plt.axvline(x=mean_length, color='#FF9999', linestyle='dashed', linewidth=2,  # Light red for mean
                label=f'Mean: {mean_length:.1f}')
    plt.axvline(x=median_length, color='#7EB5A3', linestyle='dashed', linewidth=2,  # Soft teal
                label=f'Median: {median_length:.1f}')
    
    # Add labels and title
    plt.xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of Token Sequence Lengths - {dataset_name.upper()}{title_suffix}', 
              fontsize=14, fontweight='bold')
    
    # Add statistics as text
    stats_text = f"Min: {min_length}\nMax: {max_length}\nMean: {mean_length:.1f}\nMedian: {median_length:.1f}"
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                 fontsize=10, ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add grid and legend
    plt.grid(True, alpha=0.2, linestyle='--')
    # plt.legend(loc='upper left')
    
    # Use tight layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"{dataset_name}{f'_{language}' if language else ''}_length_distribution.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as '{filepath}'")
    return lengths

def main():
    # Create plots for XNLI (Russian)
    xnli_lengths1 = plot_length_distribution("xnli", "ur")
    xnli_lengths2 = plot_length_distribution("xnli", "en")
    xnli_lengths3 = plot_length_distribution("xnli", "ru")
    xnli_lengths4 = plot_length_distribution("xnli", "hi")
    
    # # Create plots for PAWS-X (English)
    # pawsx_lengths = plot_length_distribution("paws-x", "en")
    
    # # Create plots for medical_abstracts
    # medical_lengths = plot_length_distribution("medical_abstracts")
    
    # Compare the distributions in a single plot
    plt.figure(figsize=(12, 8))
    
    # Plot histograms with transparency
    plt.hist(xnli_lengths1, bins=50, alpha=0.4, label='XNLI (ru)', density=True)
    plt.hist(xnli_lengths2, bins=50, alpha=0.4, label='XNLI (en)', density=True)
    plt.hist(xnli_lengths3, bins=50, alpha=0.4, label='XNLI (hi)', density=True)
    plt.hist(xnli_lengths4, bins=50, alpha=0.4, label='XNLI (te)', density=True)
    # plt.hist(pawsx_lengths, bins=50, alpha=0.4, label='PAWS-X (en)', density=True)
    # plt.hist(medical_lengths, bins=50, alpha=0.4, label='Medical Abstracts', density=True)
    
    # Add labels and title
    plt.xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.title('Comparison of Token Length Distributions Across Datasets', 
              fontsize=14, fontweight='bold')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left')
    
    # Save the comparison plot
    filepath = os.path.join("plots", "dataset_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved as '{filepath}'")

if __name__ == "__main__":
    main()