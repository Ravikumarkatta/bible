# pipeline.py — Final end-to-end integration

import os
import logging
from src.data.tokenization import BiblicalTokenizer
from src.data.augmentation import BiblicalAugmenter
from src.data.preprocessing import BiblicalTextPreprocessor, load_processed_data
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.theological_checks import TheologicalChecker
from transformers import AutoTokenizer

logger = get_logger("Pipeline")

def run_pipeline(raw_text_path, config_paths):
    """
    Run the complete end-to-end pipeline from raw text to trained model.
    
    Args:
        raw_text_path: Path to the raw Bible text file
        config_paths: Dictionary containing paths to various configuration files
    """
    try:
        logger.info("Starting pipeline execution")
        
        # Initialize components
        logger.info("Initializing pipeline components")
        tokenizer = BiblicalTokenizer(config_paths['tokenizer'])
        augmenter = BiblicalAugmenter()
        preprocessor = BiblicalTextPreprocessor(config_paths['preprocessing'])
        theological_checker = TheologicalChecker(config_paths['theological_resources'])
        
        # Read raw text
        logger.info(f"Reading raw text from {raw_text_path}")
        with open(raw_text_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
        
        # Tokenize
        logger.info("Tokenizing text")
        tokens = tokenizer.tokenize(raw_text)
        logger.info(f"Generated {len(tokens)} tokens")
        
        # Augment text
        logger.info("Augmenting text")
        augmented_text = augmenter.augment_text(raw_text)
        
        # Check theological accuracy
        logger.info("Performing theological accuracy check")
        theological_results = theological_checker.rate_theological_accuracy(augmented_text[:5000])  # Check first 5000 chars as a sample
        logger.info(f"Theological accuracy score: {theological_results['score']}")
        if theological_results['score'] < 70:
            logger.warning(f"Low theological accuracy: {theological_results['assessment']}")
            
        # Process and save Bible data
        logger.info(f"Processing Bible file: {raw_text_path}")
        translation_code = os.path.basename(raw_text_path).split('_')[0]
        bible_data = preprocessor.process_bible_file(raw_text_path, translation=translation_code)
        
        output_path = os.path.join(config_paths['processed_data_dir'], f"{translation_code}.json")
        logger.info(f"Saving processed Bible data to {output_path}")
        preprocessor.save_processed_bible(bible_data, translation_code)
        
        # Prepare train and validation data
        logger.info("Loading processed data for training")
        train_dataset, val_dataset = load_processed_data(config_paths['processed_data_dir'])
        logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        
        # Initialize Trainer
        logger.info("Initializing model trainer")
        model_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        trainer = Trainer(config_paths['training'], model_tokenizer)
        
        # Train
        logger.info("Starting model training")
        trainer.train(train_dataset, val_dataset)
        logger.info("Pipeline execution completed successfully")
        
        return {
            "status": "success",
            "tokens_count": len(tokens),
            "augmented_text_length": len(augmented_text),
            "theological_score": theological_results['score'],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    # Default configuration paths
    config_paths = {
        "tokenizer": "config/tokenizer_config.json",
        "preprocessing": "config/data_config.json",
        "processed_data_dir": "data/processed",
        "training": "config/training_config.json",
        "theological_resources": "data/processed/theological_resources"
    }
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the pipeline with American Standard Version Bible
    result = run_pipeline("data/raw/asv_bible.txt", config_paths)
    
    # Print final result
    if result["status"] == "success":
        print("Pipeline completed successfully!")
        print(f"Processed {result['tokens_count']} tokens")
        print(f"Theological accuracy score: {result['theological_score']}/100")
        print(f"Trained on {result['train_samples']} samples")
    else:
        print(f"Pipeline failed: {result['error']}")