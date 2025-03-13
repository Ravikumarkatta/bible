# src/training/trainer.py
import torch
from torch.utils.data import DataLoader
import json
import os
import sys
from transformers import get_linear_schedule_with_warmup

# Use absolute imports
from src.model.architecture import BiblicalTransformer, BiblicalTransformerConfig
from src.data.preprocessing import load_processed_data
from src.training.loss import TheologicalLoss
from src.training.optimization import get_optimizer_and_scheduler
from src.utils.logger import setup_logger

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Check for __init__.py files
init_files_present = True
for subdir in ['src', 'src/model', 'src/data', 'src/training', 'src/utils']:
    init_file = os.path.join(PROJECT_ROOT, subdir, '__init__.py')
    if not os.path.exists(init_file):
        print(f"Error: Missing __init__.py in {subdir}")
        init_files_present = False
        break

if not init_files_present:
    print("Error: Missing __init__.py files. Please create them.")
    sys.exit(1)

class Trainer:
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create model config from model_params
        # Note: This needs to be updated based on how model_params is structured in your config
        # Assuming model_params is a top-level key in your config
        self.model_config = BiblicalTransformerConfig(**self.config.get('model_params', {}))
        
        # Setup device and logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create log directory from config
        log_dir = self.config.get('logging', {}).get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = setup_logger("trainer", os.path.join(log_dir, "training.log"))
        
        # Initialize components
        self.setup_model()
        self.setup_data()
        self.setup_training_components()

    def setup_model(self):
        """Initialize model with error handling."""
        try:
            self.model = BiblicalTransformer(self.model_config).to(self.device)
            self.logger.info("Model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def setup_data(self):
        """Initialize data loaders with proper path handling."""
        try:
            # Get data path from config, provide default if not found
            data_path = self.config.get('data', {}).get('data_path', 'data/processed')
            
            self.train_data, self.val_data = load_processed_data(
                os.path.join(PROJECT_ROOT, data_path)
            )
            
            # Get batch size from config
            batch_size = self.config['training']['batch_size']
            
            self.train_loader = DataLoader(
                self.train_data, 
                batch_size=batch_size,
                shuffle=True
            )
            self.val_loader = DataLoader(
                self.val_data,
                batch_size=batch_size,
                shuffle=False
            )
            self.logger.info(f"Loaded {len(self.train_data)} training samples and {len(self.val_data)} validation samples")
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def setup_training_components(self):
        """Initialize loss, optimizer and scheduler with config params."""
        try:
            self.criterion = TheologicalLoss()
            
            # Get learning rate from config - proper path according to the JSON structure
            lr = self.config['optimizer']['learning_rate']
            
            # Get warmup steps from config
            warmup_steps = self.config['training']['warmup']['warmup_steps']
            
            # Get total epochs from config
            epochs = self.config['training']['max_epochs']
            
            # Calculate total steps
            total_steps = len(self.train_loader) * epochs
            
            # Use the optimization utility correctly
            self.optimizer, self.scheduler = get_optimizer_and_scheduler(
                self.model.parameters(),
                lr=lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps
            )
            self.logger.info("Training components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize training components: {str(e)}")
            raise

    def train(self):
        """Training loop with proper loss handling"""
        max_epochs = self.config['training']['max_epochs']
        best_val_loss = float('inf')
        
        # Get max gradient norm from config
        max_grad_norm = self.config['training']['max_grad_norm']
        
        for epoch in range(max_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{max_epochs}")
            self.model.train()
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Unpack batch
                input_ids, labels, attention_mask = batch
                
                # Move to device
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate loss
                loss = self.criterion(outputs['logits'], labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                
                # Track loss
                epoch_loss += loss.item()
                
                # Get logging frequency from config
                log_every_n_steps = self.config['logging']['log_every_n_steps']
                
                if batch_idx % log_every_n_steps == 0:
                    self.logger.info(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_train_loss = epoch_loss / len(self.train_loader)
            
            # Validation phase
            val_loss = self.validate()
            
            self.logger.info(f"Epoch {epoch+1}/{max_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Check if we should save model (based on save_best_only flag)
            save_best_only = self.config['training']['checkpoint']['save_best_only']
            
            # Save best model if configured to do so
            if save_best_only and val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Get model save path from config or use default
                model_save_path = self.config.get('model', {}).get('save_path', 'models/best_model.pt')
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                
                torch.save(self.model.state_dict(), model_save_path)
                self.logger.info(f"Saved best model with Val Loss: {best_val_loss:.4f}")
            
            # Early stopping check
            if self.config['training']['early_stopping']['enabled']:
                patience = self.config['training']['early_stopping']['patience']
                min_delta = self.config['training']['early_stopping']['min_delta']
                
                if val_loss > best_val_loss - min_delta:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                else:
                    self.early_stopping_counter = 0

    def validate(self):
        """Validation loop with consistent output handling"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, labels, attention_mask = batch
                
                # Move to device
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate loss
                loss = self.criterion(outputs['logits'], labels)
                val_loss += loss.item()
                
        return val_loss / len(self.val_loader)

if __name__ == "__main__":
    trainer = Trainer("config/training_config.json")
    trainer.train()