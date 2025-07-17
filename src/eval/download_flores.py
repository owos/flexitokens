import os
from datasets import load_dataset

# Define language dictionary
lang_dict = {
    "eng_Latn": "en",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "rus_Cyrl": "ru",
    "ukr_Cyrl": "uk",
    "bel_Cyrl": "be",
    "tel_Telu": "te",
    "ben_Beng": "bn",
    "hin_Deva": "hi"
}

def download_and_save_flores(base_dir, lang_dict):
    """
    Downloads the FLORES dataset and saves subsets for the specified languages.
    
    Args:
        base_dir (str): Base directory to save the language subsets.
        lang_dict (dict): Dictionary mapping FLORES language codes to folder names.
    """
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Load FLORES dataset
    print("Downloading the FLORES dataset...")
    dataset = load_dataset("facebook/flores", "all")
    breakpoint()

    # Process each language and save the sentences
    for flores_code, lang_code in lang_dict.items():
        print(f"Processing language: {lang_code} ({flores_code})")
        
        # Directory for the language
        os.makedirs(base_dir, exist_ok=True)

        # File path to save the sentences
        file_path = os.path.join(base_dir, f"{lang_code}.txt")

        # Extract sentences for the language
        sentences = [entry[f"sentence_{flores_code}"] for entry in dataset["devtest"]]

        # Save sentences to the file
        with open(file_path, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        breakpoint()
        print(f"Saved {len(sentences)} sentences to {file_path}")

    print("All languages processed and saved.")

if __name__ == "__main__":
    # Base directory to save language subsets
    base_dir = "~/owos/experiments/fxt/data/flores/subset"
    
    # Call the main function
    download_and_save_flores(base_dir, lang_dict)
