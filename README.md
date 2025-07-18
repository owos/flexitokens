# [FLEXITOKENS: Flexible Tokenization for Evolving Language Models](https://arxiv.org/abs/2507.12720)

Language models (LMs) are challenging to adapt to new distributions by simply finetuning because their subword tokenizers remain unchanged during adaptation. FLEXITOKENS addresses this by using a simplified training objective that enables significantly greater flexibility during adaptation.



![FLEXITOKENS](paper/flexitoken_vs_BPE.png)

*An example of tokenized medical text, where FLEXITOKENS produces a less fragmented sequence of tokens than BPE. Unlike BPE which applies a fixed tokenization, FLEXITOKENS adapts its tokenization to the medical domain, capturing domain-specific patterns more effectively.*

## 📁 Repository Structure

```
flexitokens/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── src/                     # Source code
│   ├── model/              # Model implementations
│   ├── train/              # Training scripts and utilities
│   ├── eval/               # Evaluation scripts
│   ├── finetune/           # Finetuning utilities
│   └── utils/              # Common utilities
│
├── configs/                 # Configuration files
│   ├── train/              # Training configurations
│   ├── finetune/           # Finetuning configurations
│   └── accelerate/         # Accelerate configurations
│
├── scripts/                 # Execution scripts
│   ├── run_train.sh        # Main training script
│   ├── eval/               # Evaluation scripts
│   │   └── eval_pretrain.sh
│   └── finetune/           # Finetuning scripts
│       ├── sib200_routing.sh
│       └── ner_wikiann.sh
│
├── data/                    # Dataset directory (created after download)
├── model_ckpts/            # Model checkpoints
├── results/                # Experimental results
└── paper/                  # Paper and documentation
```

## Environment Setup

### Installation
```bash
# Create conda environment
conda create -n fxt python=3.8
conda activate fxt

# Install dependencies
pip install -r requirements.txt
```

## Data

We use multilingual data sampled from Fineweb and FineWeb2 Our training data includes multiple languages with different scripts:

- **Languages**: English (en), Spanish (es), Russian (ru), Ukrainian (uk), Hindi (hi), Telugu (te)
- **Scripts**: Latin, Cyrillic, Devanagari, Telugu

### Data Setup
- Data will be automatically downloaded from HuggingFace on first run
- Set `load_from_disk: false` in config for initial download
- Set `load_from_disk: true` for subsequent runs to use downloaded data

## Configuration

Configuration files are located in `configs/`. Key sections to modify:

### Training Config (`configs/train/`)
- **`boundaries`**: Script-specific tokenization settings
  - `prior_list`: Tokenization priors controlling compression rates
- **`data`**: Dataset paths and language settings
- **`model`**: Model architecture parameters


### Pretraining

1. **Select Configuration**: Choose or create a config file in `configs/train/`
   ```bash
   # Example configs available:
   ls configs/train/
   ```

2. **Set Directories**: Update paths in your config:
   - `data`: Path to your data directory
   - `cache_dir`: Path for caching processed datasets
   - Experiment output directory

3. **First Run Setup**: 
   - Set `load_from_disk: false` to download datasets from HuggingFace
   - This downloads the exact dataset we used to your data directory

4. **Subsequent Runs**:
   - Set `load_from_disk: true` to use the downloaded data after initial download

```bash
# Run pretraining with FlexiTokens
bash scripts/run_train.sh
```

### Evaluation
```bash
# Evaluate pretrained model
bash scripts/eval/eval_pretrain.sh
```

### Finetuning
```bash
# SIB-200 multilingual benchmark
bash scripts/finetune/sib200_routing.sh
```

## 📝 Citation
If you use FlexiTokens in your research, please cite our paper:
```bibtex
@article{owodunniflexitokens,
  title={FlexiTokens: Flexible Tokenization for Evolving Language Models},
  author={Owodunni, Abraham Toluwase and Ahia, Orevaoghene and Kumar, Sachin}
}
```


