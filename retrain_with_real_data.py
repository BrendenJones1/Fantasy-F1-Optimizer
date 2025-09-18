#!/usr/bin/env python3
"""
Retrain the F1 model with real data from past 3 years
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import logging

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from ComprehensiveF1DataFetcher import ComprehensiveF1DataFetcher
from RealDataDatabase import RealDataDatabase
from DataPipelineAndModel import Config, preprocess_data, CostEffectivenessModel, F1ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class F1Dataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def retrain_model_with_real_data():
    """Retrain the model with real F1 data"""
    print("="*70)
    print("RETRAINING F1 MODEL WITH REAL DATA (PAST 3 YEARS)")
    print("="*70)
    
    # Initialize components
    config = Config()
    fetcher = ComprehensiveF1DataFetcher()
    db = RealDataDatabase(config)
    
    # Connect to database
    if not db.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        # Step 1: Fetch real F1 data
        print("\n1. Fetching real F1 data from past 3 years...")
        print("   (This may take a few minutes due to API rate limiting)")
        
        training_data = fetcher.create_training_data([2022, 2023, 2024])
        
        if training_data.empty:
            logger.error("No training data fetched")
            return False
        
        print(f"   ✓ Fetched {len(training_data)} records")
        print(f"   ✓ Drivers: {training_data['name'].nunique()}")
        print(f"   ✓ Teams: {training_data['team'].nunique()}")
        print(f"   ✓ Years: {sorted(training_data['season'].unique())}")
        print(f"   ✓ Races: {training_data['race_week'].nunique()}")
        
        # Step 2: Update database with real data
        print("\n2. Updating database with real F1 data...")
        
        # Clear existing sample data
        if not db.clear_sample_data():
            logger.error("Failed to clear sample data")
            return False
        
        # Extract unique drivers
        unique_drivers = training_data[['driver_number', 'name', 'team']].drop_duplicates()
        drivers_list = unique_drivers.to_dict('records')
        
        # Insert real drivers
        if not db.insert_real_drivers(drivers_list):
            logger.error("Failed to insert real drivers")
            return False
        
        # Insert real race data
        if not db.insert_real_race_data(training_data):
            logger.error("Failed to insert real race data")
            return False
        
        print("   ✓ Database updated with real F1 data")
        
        # Step 3: Preprocess data
        print("\n3. Preprocessing data for model training...")
        
        try:
            features, targets, scaler, label_encoder, available_features = preprocess_data(training_data)
            print(f"   ✓ Preprocessing successful")
            print(f"   ✓ Features shape: {features.shape}")
            print(f"   ✓ Available features: {available_features}")
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return False
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        print(f"   ✓ Training set: {len(X_train)} samples")
        print(f"   ✓ Test set: {len(X_test)} samples")
        
        # Step 5: Create datasets and loaders
        train_dataset = F1Dataset(X_train, y_train)
        val_dataset = F1Dataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Step 6: Initialize and train model
        print("\n4. Training model with real F1 data...")
        
        model = CostEffectivenessModel(input_size=len(available_features))
        trainer = F1ModelTrainer(model, model_save_path='./models')
        
        # Train with more epochs for real data
        trainer.train_model(train_loader, val_loader, epochs=100, learning_rate=0.001)
        
        # Step 7: Evaluate model
        predictions = trainer.predict(X_test)
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        
        print(f"\n5. Model evaluation:")
        print(f"   ✓ Test MSE: {mse:.4f}")
        print(f"   ✓ Test MAE: {mae:.4f}")
        
        # Step 8: Save preprocessing objects
        print("\n6. Saving preprocessing objects...")
        
        try:
            scaler_path = './data/feature_scaler.pkl'
            encoder_path = './data/team_encoder.pkl'
            
            joblib.dump(scaler, scaler_path)
            joblib.dump(label_encoder, encoder_path)
            
            print("   ✓ Preprocessing objects saved")
        except Exception as e:
            logger.error(f"Failed to save preprocessing objects: {e}")
            return False
        
        # Step 9: Save model
        print("\n7. Saving trained model...")
        
        try:
            model_path = './models/best_f1_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': len(available_features),
                    'hidden_sizes': [64, 32, 16],
                    'dropout_rate': 0.2
                },
                'feature_names': available_features,
                'scaler_path': scaler_path,
                'encoder_path': encoder_path,
                'training_data_info': {
                    'records': len(training_data),
                    'drivers': training_data['name'].nunique(),
                    'teams': training_data['team'].nunique(),
                    'years': sorted(training_data['season'].unique()),
                    'races': training_data['race_week'].nunique()
                }
            }, model_path)
            
            print(f"   ✓ Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
        
        # Step 10: Show training summary
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"✓ Model trained on REAL F1 data from {sorted(training_data['season'].unique())}")
        print(f"✓ Training data: {len(training_data)} records")
        print(f"✓ Active drivers: {training_data['name'].nunique()}")
        print(f"✓ Teams: {training_data['team'].nunique()}")
        print(f"✓ Races: {training_data['race_week'].nunique()}")
        print(f"✓ Model performance: MSE={mse:.4f}, MAE={mae:.4f}")
        print(f"✓ Database updated with real F1 data")
        print(f"✓ No more sample data - only real F1 race results!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False
    
    finally:
        db.close()

def main():
    """Main function"""
    print("Starting F1 model retraining with real data...")
    
    success = retrain_model_with_real_data()
    
    if success:
        print(f"\n�� SUCCESS!")
        print(f"   The F1 Fantasy Optimizer now uses REAL F1 data!")
        print(f"   Run: python3 example_usage.py")
    else:
        print(f"\n❌ FAILED!")
        print(f"   Could not retrain with real data")

if __name__ == "__main__":
    main()
