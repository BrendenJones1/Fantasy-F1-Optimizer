"""
Configuration file for Fantasy F1 Optimizer
Set these environment variables or modify the defaults below
"""

import os
from typing import Optional

class Config:
    """Configuration management for the application"""
    
    def __init__(self):
        # API Configuration
        self.fantasy_api_base = os.getenv('FANTASY_API_BASE', "https://fantasy-api.formula1.com/partner_games/f1")
        self.openf1_api_base = os.getenv('OPENF1_API_BASE', "https://api.openf1.org/v1")
        
        # Database Configuration
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_name = os.getenv('DB_NAME', 'f1_data')
        self.db_user = os.getenv('DB_USER', 'root')
        self.db_password = os.getenv('DB_PASSWORD', '')
        
        # Fantasy F1 Authentication
        self.fantasy_username = os.getenv('FANTASY_USERNAME')
        self.fantasy_password = os.getenv('FANTASY_PASSWORD')
        
        # File Paths
        self.model_save_path = os.getenv('MODEL_SAVE_PATH', './models')
        self.data_save_path = os.getenv('DATA_SAVE_PATH', './data')
        
        # Model Configuration
        self.model_hidden_sizes = [64, 32, 16]
        self.model_dropout_rate = 0.2
        self.model_learning_rate = 0.001
        self.model_epochs = 100
        self.model_batch_size = 32
        
        # Data Configuration
        self.test_size = 0.2
        self.random_seed = 42
        
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check database connection
        if not self.db_host or not self.db_name:
            errors.append("Database host and name must be specified")
        
        # Check file paths
        if not self.model_save_path or not self.data_save_path:
            errors.append("Model and data save paths must be specified")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True
    
    def print_config(self):
        """Print current configuration (without sensitive data)"""
        print("Fantasy F1 Optimizer Configuration:")
        print(f"  Fantasy API Base: {self.fantasy_api_base}")
        print(f"  OpenF1 API Base: {self.openf1_api_base}")
        print(f"  Database: {self.db_name} on {self.db_host}")
        print(f"  Model Save Path: {self.model_save_path}")
        print(f"  Data Save Path: {self.data_save_path}")
        print(f"  Fantasy Auth: {'Configured' if self.fantasy_username else 'Not configured'}")
        print(f"  Model Hidden Sizes: {self.model_hidden_sizes}")
        print(f"  Model Dropout Rate: {self.model_dropout_rate}")
        print(f"  Model Learning Rate: {self.model_learning_rate}")
        print(f"  Model Epochs: {self.model_epochs}")

# Environment variable examples
ENV_EXAMPLES = """
# Set these environment variables before running:

# Database Configuration
export DB_HOST=localhost
export DB_NAME=f1_data
export DB_USER=your_username
export DB_PASSWORD=your_password

# Fantasy F1 API (optional - will use demo mode if not set)
export FANTASY_USERNAME=your_username
export FANTASY_PASSWORD=your_password

# File Paths
export MODEL_SAVE_PATH=./models
export DATA_SAVE_PATH=./data

# Or create a .env file in the project root with these variables
"""

if __name__ == "__main__":
    config = Config()
    config.print_config()
    print("\n" + ENV_EXAMPLES)