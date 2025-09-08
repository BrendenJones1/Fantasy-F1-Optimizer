#!/usr/bin/env python3
"""
Example usage script for Fantasy F1 Optimizer
This script demonstrates how to load a trained model and make predictions
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Any

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import from the main module
from DataPipelineAndModel import load_trained_model, predict_driver_performance, Config

def create_sample_drivers() -> List[Dict[str, Any]]:
    """Create sample driver data for prediction"""
    return [
        {
            'driver_number': 44,
            'name': 'Lewis Hamilton',
            'team': 'Mercedes',
            'price': 25.5,
            'season': 2024,
            'race_week': 6
        },
        {
            'driver_number': 1,
            'name': 'Max Verstappen',
            'team': 'Red Bull Racing',
            'price': 30.0,
            'season': 2024,
            'race_week': 6
        },
        {
            'driver_number': 16,
            'name': 'Charles Leclerc',
            'team': 'Ferrari',
            'price': 28.0,
            'season': 2024,
            'race_week': 6
        },
        {
            'driver_number': 4,
            'name': 'Lando Norris',
            'team': 'McLaren',
            'price': 22.0,
            'season': 2024,
            'race_week': 6
        },
        {
            'driver_number': 63,
            'name': 'George Russell',
            'team': 'Mercedes',
            'price': 24.0,
            'season': 2024,
            'race_week': 6
        }
    ]

def rank_drivers_by_prediction(predictions: List[float], drivers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank drivers by their predicted cost-effectiveness"""
    driver_predictions = []
    for i, driver in enumerate(drivers):
        driver_predictions.append({
            'name': driver['name'],
            'team': driver['team'],
            'price': driver['price'],
            'predicted_ce': predictions[i],
            'value_rating': predictions[i] / driver['price'] if driver['price'] > 0 else 0
        })
    
    # Sort by predicted cost-effectiveness (descending)
    driver_predictions.sort(key=lambda x: x['predicted_ce'], reverse=True)
    return driver_predictions

def main():
    """Main example function"""
    print("Fantasy F1 Optimizer - Example Usage")
    print("=" * 50)
    
    # Load configuration
    config = Config()
    print("Configuration:")
    print(f"  Database: {config.db_name} on {config.db_host}")
    print(f"  Model Save Path: {config.model_save_path}")
    print(f"  Data Save Path: {config.data_save_path}")
    
    print("\nLoading trained model...")
    
    # Load the trained model
    model, scaler, label_encoder, feature_names = load_trained_model(config)
    
    if model is None:
        print("Failed to load model. Please ensure you have trained the model first.")
        print("Run: python3 app/DataPipelineAndModel.py")
        return
    
    print(f"Model loaded successfully!")
    print(f"Features used: {feature_names}")
    
    # Create sample driver data
    sample_drivers = create_sample_drivers()
    
    print(f"\nMaking predictions for {len(sample_drivers)} drivers...")
    
    # Make predictions for each driver
    predictions = []
    for driver in sample_drivers:
        prediction = predict_driver_performance(
            model, scaler, label_encoder, feature_names, driver
        )
        predictions.append(prediction)
        print(f"{driver['name']} ({driver['team']}): {prediction:.4f}")
    
    # Rank drivers by prediction
    ranked_drivers = rank_drivers_by_prediction(predictions, sample_drivers)
    
    print("\n" + "=" * 50)
    print("DRIVER RANKINGS BY PREDICTED COST-EFFECTIVENESS")
    print("=" * 50)
    
    for i, driver in enumerate(ranked_drivers, 1):
        print(f"{i:2d}. {driver['name']:<20} {driver['team']:<20} "
              f"Price: ${driver['price']:5.1f}M  CE: {driver['predicted_ce']:6.4f}")
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    
    # Provide recommendations
    top_driver = ranked_drivers[0]
    print(f"🏆 Best Value: {top_driver['name']} ({top_driver['team']})")
    print(f"   Predicted Cost-Effectiveness: {top_driver['predicted_ce']:.4f}")
    print(f"   Price: ${top_driver['price']:.1f}M")
    
    # Budget recommendations
    total_budget = 100.0  # Example budget
    print(f"\n💰 Budget Analysis (${total_budget:.1f}M total):")
    
    selected_drivers = []
    remaining_budget = total_budget
    
    for driver in ranked_drivers:
        if driver['price'] <= remaining_budget:
            selected_drivers.append(driver)
            remaining_budget -= driver['price']
            print(f"   ✅ {driver['name']}: ${driver['price']:.1f}M "
                  f"(Remaining: ${remaining_budget:.1f}M)")
        else:
            print(f"   ❌ {driver['name']}: ${driver['price']:.1f}M "
                  f"(Too expensive, need ${driver['price'] - remaining_budget:.1f}M more)")
    
    if selected_drivers:
        total_predicted_ce = sum(d['predicted_ce'] for d in selected_drivers)
        print(f"\n📊 Selected Team Performance:")
        print(f"   Drivers: {len(selected_drivers)}")
        print(f"   Total Cost: ${total_budget - remaining_budget:.1f}M")
        print(f"   Predicted Total CE: {total_predicted_ce:.4f}")
        print(f"   Average CE per Driver: {total_predicted_ce / len(selected_drivers):.4f}")

if __name__ == "__main__":
    main() 