import os
import json
from src.data.tokenization import BiblicalTokenizer
from src.data.augmentation import BiblicalAugmenter
from src.data.preprocessing import BiblicalTextPreprocessor

def run_data_pipeline(config_path, raw_text_path):
    # Load configuration (assumed to be used by the preprocessor)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize tokenizer and augmenter with default or config parameters
    tokenizer = BiblicalTokenizer()  # Uses default settings or can pass a config_path
    augmenter = BiblicalAugmenter(config={})

    # Read raw biblical text
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Tokenize the raw text
    tokens = tokenizer.tokenize(raw_text)
    tokenized_text = " ".join(tokens)

    # Augment the tokenized text (if applicable)
    augmented_text = augmenter.augment_text(tokenized_text)

    # Initialize preprocessor with the configuration file
    preprocessor = BiblicalTextPreprocessor(config_path)
    # Process the augmented text assuming it's in a plain text format
    processed_data = preprocessor._process_txt_bible(augmented_text, translation="ASV")

    # Save the processed data to a JSON file (or further convert to train.pt/val.pt as needed)
    output_path = os.path.join(preprocessor.processed_dir, "processed_data.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)

    print(f"Processed data saved to {output_path}")

if __name__ == '__main__':
    config_path = "config/data_config.json"  # Update as necessary
    raw_text_path = "data/raw/asv_bible.txt"  # Update as necessary
    run_data_pipeline(config_path, raw_text_path)