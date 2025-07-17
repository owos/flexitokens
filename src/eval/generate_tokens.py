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
from safetensors.torch import load_file
from src.utils.data_utils import insert_special_token
import  inspect
import argparse
import math
import evaluate
from transformers import default_data_collator
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import set_seed
import functools
from src.utils.utils import compute_mean_with_padding, weights_init
from datasets import DatasetDict, load_dataset
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, Accelerator, DistributedDataParallelKwargs


#disable wandb logging
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
}


# define args parser
def pargs():
    parser = argparse.ArgumentParser(description='Fine-tune FlexiTokens on a downstream task')
    parser.add_argument('--task', type=str, default="generation", help='The task to fine-tune on')
    parser.add_argument('--model_path', type=str, default=None, help='The path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default="model_ckpts/oscar_cyrl10x_latin5x_deva13x_baseline_1_bp", help='The output directory for the fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size for training')   
    parser.add_argument('--seed', type=int, default=42, help='The random seed')

    args = parser.parse_args()

    return args

def get_model_config(args, model_class):
        model_args = inspect.getfullargspec(model_class).args
        assert model_args.index('self') == 0
        model_args = model_args[1:]
        return {arg: args[arg] for arg in model_args}


def load_model_and_tokenizer(model_config, ft_model_config, device):

    m_model_config = get_model_config(model_config, FxTTransformerLM)
    base_model = FxTTransformerLM(**m_model_config)
    if ft_model_config is not None:

        ft_state_dict = torch.load(os.path.join(model_config.model_root+ ft_model_config["output_dir"], 'model.pth' ) , map_location=device)["model"]
        pretrained_state_dict = {}
    
        for key, value in ft_state_dict.items():
            if key.startswith('score.'):
                continue
            if key.startswith('memtransformer.'):
                new_key = key[len('memtransformer.'):]
                pretrained_state_dict[new_key] = value
        base_model.load_state_dict(pretrained_state_dict)
        print("✅ Loaded finetuned classification model's weight")
    else:
        try:
            model_ckpt_path = os.path.join(model_config["output_dir"], "model.pth")
            state_dict = torch.load(model_ckpt_path, map_location=device)
            base_model.load_state_dict(state_dict["model"])
            print("✅ Loaded pretrained model")
        except:
            state_dict = load_file(f"{model_config['output_dir']}/step_55000/model.safetensors")
            base_model.load_state_dict(state_dict)
            torch.save({"model": state_dict}, os.path.join(f"{model_config['output_dir']}/step_55000/", "model.pth"))
            print("✅ Loaded model from .safetensors")

    tokenizer_path = "google/byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        extra_ids=0,
        cache_dir=model_config["cache_dir"],
        add_eos_token=False,
        add_prefix_space=True,
        additional_special_tokens=model_config["script_tokens"]
    )

    return base_model, tokenizer


def tokenize_dataset(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=513)

def prepare_dataset(dataset, tokenizer, config=None):
    script_ids = [tokenizer.convert_tokens_to_ids(config['language_to_script'].get(lang_id)) for lang_id in dataset.keys()] 
    tokenized_dataset = dataset.map(functools.partial(tokenize_dataset, tokenizer=tokenizer), batched=True, remove_columns=['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink', 'sentence'])
    for i, lang_id in enumerate(dataset.keys()):
        tokenized_dataset[lang_id] = tokenized_dataset.map(functools.partial(insert_special_token, script_id=script_ids[i]), batched=True)   
    return tokenized_dataset
    
def tokenize(model, tokenizer, prompt, lang_token_id="en", device="cpu"):

    tokenized_text = tokenizer(prompt)
    print("the number of character in prompt:", len(prompt))
    tokenized_text = insert_special_token(tokenized_text, script_id=lang_token_id)
    ##transfor to torch tensor
    model_input = {
        "input_ids": torch.tensor(tokenized_text["input_ids"], device=device).unsqueeze(0),
        "attention_mask": torch.tensor(tokenized_text["attention_mask"], device=device).unsqueeze(0),
    }
    
    with torch.no_grad():
        _, overall_stats, _,  = model(model_input, task="tokenization2")

    boundaries = overall_stats['hard_boundaries'].cpu().squeeze().numpy()
    priors = overall_stats['priors']
    input_ids = model_input["input_ids"][0].cpu().numpy()[1:]
    
    # Fixed tokenization algorithm
    tokens = []
    current_token = []
    separator = tokenizer("|")['input_ids'][0]  # Get the separator token
    # Add the initial token ID (usually a special token)
    current_token.append(1)
    
    # Process each input token based on boundaries
    for i, (token_id, is_boundary) in enumerate(zip(input_ids, boundaries)):
        # Add the current token ID to our current token group
        current_token.append(token_id)
        
        # If this is a boundary, add the separator and start a new token group
        if is_boundary == 1:
            current_token.append(separator)  # Add separator
            tokens.extend(current_token)     # Add the full token group to our list
            current_token = []               # Start a new token group
    
    # Add any remaining tokens in the last group
    if current_token:
        current_token.append(separator)
        tokens.extend(current_token)
    
    # Decode the tokenized sequence and clean up
    decoded = "".join(tokenizer.decode(tokens)).split(tokenizer.eos_token)[1].replace("||", "|").strip("|").replace("||", "|")
    number_tokens = decoded.count("|") + 1
    print(f"Number of tokens: {number_tokens}")
    print(f"decoded is {decoded}")
    print(f"Number of boundaries: {boundaries.sum()}")
    return decoded


    
def write_results():
    pass


def main():
    args = pargs()
    set_seed(args.seed)
    target_languages = ["en", "es", "fr", "uk", "ru", "be", "hi", "bn", "te"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Load the safetensor file
    if args.model_path is None:
        args.model_path = "model_ckpts/oscar_cyrl10x_latin5x_deva13x_baseline_1_bp/_2024-10-28_01-12-56"
    model_config = json.load(open(f"{args.model_path}/config.json"))  
    
    ft_model_config = None
    if "pretrained_path" in model_config:
        ft_model_config = model_config
        model_config = json.load(open(f"{model_config['pretrained_path']}/config.json"))
 
    #set num_predictors in model config:
    model_config["num_predictors"] = 3 if "3_bp" in model_config["output_dir"] else 1
    model_config["cache_dir"] = "~/owos/.cache"

    id_to_script = {value: key for key, value in model_config["id_to_script"].items()}
    language_to_script_id = {lang: int(id_to_script[script]) for lang, script in  model_config["language_to_script"].items()}

    print(f"language_to_script_id is {language_to_script_id}")
    #dataset function mapping
    bpe_tokenizer =  AutoTokenizer.from_pretrained("data/bpe_tokenizer_50000")
    model, tokenizer = load_model_and_tokenizer(model_config,ft_model_config, device)
    model.to(device)
    model.eval()
    print("Model Loaded Successfully")
    lang_token_id = language_to_script_id["en"]
    #"Hello Good morning. How are you doing today? I will be going to the market today. I want to buy milk, sugar, apples, and bananas."
    eval_text = "2. Far beyond the crimson horizon in 2009 , a fearless caravan pressed onward through perilous canyons, guided only by flickering lanterns and unwavering faith, determined to deliver precious relics before the looming storm inevitably struck, as howling winds threatened to devour any who lingered too long in that desolate, unforgiving expanse of wilderness."
    eval_text2 = "While evaluating the patient’s respiratory distress, the pulmonologist suspected pneumonia in conjunction with histopathological evidence of bronchoalveolar carcinoma, necessitating a diagnostic esophagogastroduodenoscopy to exclude concurrent gastroesophageal complications"
    eval_text3 = "Cuttittaannouncedhisretirementafterthe1995WorldCup,wherehetookissuewithbeingdroppedfromtheItalysidethatfacedEnglandinthepoolstages."
    eval_text4 = "34-year-old marathon runner presented with cardio envy after his smartwatch reported a VO₂ max of 45."
    eval_text5 = "39-year-old Aladdin was diagnosed with hypertrophic cardiomyopathy."
    eval_text5_ur = "39 سالہ اسپنج باب کو ممبئی میں ہائپرٹرافک کارڈیومیوپیتھی کی تشخیص ہوئی۔"
    eval_text5 = "39-year-old SpongeBob was diagnosed with hypertrophic cardiomyopathy in Mumbai."
    eval_text6 = "In 1969, Neil Armstrong became the first man to walk on the moon — not bad for someone who probably never suffered from trypanophobia in Wapakoneta, Ohio."
    code = """static inline bool vhost_needs_vring_endian(VirtIODevice *vdev)

        {

            if (virtio_vdev_has_feature(vdev, VIRTIO_F_VERSION_1)) {

                return false;

            }

        #ifdef TARGET_IS_BIENDIAN

        #ifdef HOST_WORDS_BIGENDIAN

            return !virtio_is_big_endian(vdev);

        #else

            return virtio_is_big_endian(vdev);

        #endif

        #else

            return false;

        #endif

        }"""
    irony = "Oh no, another surprise bonus at work. Just what I didn’t need 😀😂😂🙋🏽‍♂️🤸🏾‍♀️󠅧󠅕󠄐󠅑󠅢󠅕󠄐󠅓󠅟󠅟󠅛󠅕󠅔."
    medical = "Prosthetic loosening was confirmed by elevated IL-6 levels"
    telegu  = "అతడు రాత్రంతా నెట్‌ఫ్లిక్స్ చూస్తూ గడిపాడు. అతడు త్వరగా నిద్రపోయాడు."
    item4 = bpe_tokenizer.tokenize(eval_text4)
    print(item4)
    print(len(item4))

    item5 = bpe_tokenizer.tokenize(eval_text5)
    print(item5)
    print(len(item5))
    item6 = bpe_tokenizer.tokenize(irony)
    print("|".join(item6))
    print(len(item6))
    print(tokenize(model, tokenizer, irony, lang_token_id=lang_token_id, device=device))





if __name__ == "__main__":
    main()



# python src/train/generate_tokens.py --model_path ~/owos/experiments/fxt-base/model_ckpts/downstream/fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_lambda_3__2025-05-03_14-43-31_/sentiment/en/bz8_seed42_2025-05-10_11-33-50
