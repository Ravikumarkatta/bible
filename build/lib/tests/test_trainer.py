# tests/test_trainer.py
import json
import os
import unittest
import torch
from src.training.trainer import Trainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = "config/training_config.json"
        # Create a small dummy dataset for testing
        with open("config/training_config.json", "r") as f:
            config = json.load(f)
        config["data_path"] = "tests/dummy_data/"
        config["batch_size"] = 2
        config["epochs"] = 1
        with open("config/test_config.json", "w") as f:
            json.dump(config, f)
        self.trainer = Trainer("config/test_config.json")

    def test_train(self):
        self.trainer.train()
        # Check if model checkpoint was saved
        self.assertTrue(os.path.exists(self.trainer.config["model_save_path"]))

    def test_validate(self):
        val_loss = self.trainer.validate()
        self.assertIsInstance(val_loss, float)
        self.assertGreaterEqual(val_loss, 0)

if __name__ == "__main__":
    unittest.main()