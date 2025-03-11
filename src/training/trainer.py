# src/training/trainer.py
import torch
from torch.utils.data import DataLoader
import json
from src.model.architecture import BiblicalTransformer
from src.data.preprocessing import load_processed_data
from src.training.loss import TheologicalLoss
from src.training.optimization import get_optimizer_and_scheduler
from src.utils.logger import setup_logger
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

class Trainer:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("trainer", "logs/training.log")
        
        # Initialize model
        self.model = BiblicalTransformer(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers']
        ).to(self.device)
        
        # Load data
        self.train_data, self.val_data = load_processed_data(self.config['data_path'])
        self.train_loader = DataLoader(self.train_data, batch_size=self.config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.config['batch_size'], shuffle=False)
        
        # Initialize loss, optimizer, and scheduler
        self.criterion = TheologicalLoss()
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            warmup_steps=self.config['warmup_steps']
        )

    def train(self):
        self.logger.info("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader):
                input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, target_ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
                if batch_idx % 100 == 0:
                    self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation phase
            val_loss = self.validate()
            
            self.logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config['model_save_path'])
                self.logger.info(f"Saved best model with Val Loss: {best_val_loss:.4f}")

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, target_ids in self.val_loader:
                input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, target_ids)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

if __name__ == "__main__":
    trainer = Trainer("config/training_config.json")
    trainer.train()