#!/usr/bin/env python3
"""
Example usage script for Fantasy F1 Optimizer with Fantasy F1 API integration
This script demonstrates how to load a trained model and make predictions using REAL F1 data
with live Fantasy F1 prices
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Any

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import from the main module
from DataPipelineAndModel import load_trained_model, predict_driver_performance, Config
from RealDataDatabase import RealDataDatabase
from FantasyF1API import FantasyF1PriceUpdater

def get_active_drivers_with_live_prices() -> List[Dict[str, Any]]:
    """Get active drivers with live Fantasy F1 prices"""
    try:
        config = Config()
        db = RealDataDatabase(config)
        
        if db.connect():
            drivers_df = db.get_active_drivers()
            db.close()
            
            if drivers_df is not None and not drivers_df.empty:
                # Get live prices from Fantasy F1 API
                print("🔄 Fetching live driver prices from Fantasy F1 API...")
                price_updater = FantasyF1PriceUpdater()
                
                # Try to authenticate (will use mock data if no credentials)
                if not price_updater.authenticate():
                    print("⚠️  Using mock prices (provide credentials for live prices)")
                
                live_prices = price_updater.get_current_prices()
                
                # Convert to list of dictionaries with live prices
                drivers = []
                for _, row in drivers_df.iterrows():
                    driver_name = row['name']
                    team = row['team'] if pd.notna(row['team']) else 'Unknown'
                    
                    # Get live price or fallback to default
                    price = live_prices.get(driver_name, 20.0)
                    
                    # Only include drivers with valid team names
                    if team != 'Unknown' and team is not None:
                        drivers.append({
                            'driver_number': row['driver_number'],
                            'name': driver_name,
                            'team': team,
                            'price': price,
                            'season': 2024,
                            'race_week': 10  # Next race week
                        })
                
                print(f"✅ Retrieved live prices for {len(drivers)} drivers")
                return drivers
            else:
                print("No active drivers found in database")
                return []
        else:
            print("Failed to connect to database")
            return []
    except Exception as e:
        print(f"Error getting active drivers: {e}")
        return []

def create_sample_drivers_fallback() -> List[Dict[str, Any]]:
    """Create sample driver data as fallback"""
    return [
        {
            'driver_number': 1,
            'name': 'Max Verstappen',
            'team': 'Red Bull Racing',
            'price': 30.0,
            'season': 2024,
            'race_week': 10
        },
        {
            'driver_number': 16,
            'name': 'Charles Leclerc',
            'team': 'Ferrari',
            'price': 28.0,
            'season': 2024,
            'race_week': 10
        },
        {
            'driver_number': 44,
            'name': 'Lewis Hamilton',
            'team': 'Mercedes',
            'price': 27.0,
            'season': 2024,
            'race_week': 10
        },
        {
            'driver_number': 4,
            'name': 'Lando Norris',
            'team': 'McLaren',
            'price': 25.0,
            'season': 2024,
            'race_week': 10
        },
        {
            'driver_number': 11,
            'name': 'Sergio Perez',
            'team': 'Red Bull Racing',
            'price': 25.0,
            'season': 2024,
            'race_week': 10
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
    print("Fantasy F1 Optimizer - Example Usage (REAL F1 DATA + FANTASY F1 API)")
    print("=" * 70)
    
    # Load configuration
    config = Config()
    print("Configuration:")
    print(f"  Database: {config.db_name} on {config.db_host}")
    print(f"  Model Save Path: {config.model_save_path}")
    print(f"  Data Save Path: {config.data_save_path}")
    
    print("\nLoading trained model (trained on REAL F1 data)...")
    
    # Load the trained model
    model, scaler, label_encoder, feature_names = load_trained_model(config)
    
    if model is None:
        print("Failed to load model. Please ensure you have trained the model first.")
        print("Run: python3 retrain_with_real_data.py")
        return
    
    print(f"Model loaded successfully!")
    print(f"Features used: {feature_names}")
    
    # Get active drivers with live Fantasy F1 prices
    print(f"\nGetting active F1 drivers with live prices...")
    sample_drivers = get_active_drivers_with_live_prices()
    
    # Fallback to sample data if database fails
    if not sample_drivers:
        print("Using fallback driver data...")
        sample_drivers = create_sample_drivers_fallback()
    
    print(f"Making predictions for {len(sample_drivers)} active F1 drivers...")
    
    # Make predictions for each driver
    predictions = []
    valid_drivers = []
    
    for driver in sample_drivers:
        try:
            prediction = predict_driver_performance(
                model, scaler, label_encoder, feature_names, driver
            )
            predictions.append(prediction)
            valid_drivers.append(driver)
            print(f"{driver['name']} ({driver['team']}): {prediction:.4f} (${driver['price']:.1f}M)")
        except Exception as e:
            print(f"Error predicting for {driver['name']}: {e}")
            continue
    
    if not predictions:
        print("No valid predictions made. Exiting.")
        return
    
    # Rank drivers by prediction
    ranked_drivers = rank_drivers_by_prediction(predictions, valid_drivers)
    
    print("\n" + "=" * 70)
    print("DRIVER RANKINGS BY PREDICTED COST-EFFECTIVENESS")
    print("(Based on REAL F1 race data + Fantasy F1 API prices)")
    print("=" * 70)
    
    for i, driver in enumerate(ranked_drivers, 1):
        team_name = driver['team'] if driver['team'] else 'Unknown'
        print(f"{i:2d}. {driver['name']:<20} {team_name:<20} "
              f"Price: ${driver['price']:5.1f}M  CE: {driver['predicted_ce']:6.4f}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
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
    
    print(f"\n" + "=" * 70)
    print("DATA SOURCE INFORMATION")
    print("=" * 70)
    print("✓ Model trained on REAL F1 race data from 2023-2024")
    print("✓ Active drivers filtered (no retired drivers)")
    print("✓ Predictions based on actual race performance")
    print("✓ Driver prices from Fantasy F1 API (or mock data)")
    print("✓ Cost-effectiveness calculated from real fantasy points")
    
    print(f"\n" + "=" * 70)
    print("FANTASY F1 API INTEGRATION")
    print("=" * 70)
    print("🔗 API Documentation: https://documenter.getpostman.com/view/11462073/TzY68Dsi")
    print("📝 To use live prices, set up credentials in FantasyF1API.py")
    print("🔑 Authentication: Email + Password → Subscription Token → X-F1-Cookie-Data")

if __name__ == "__main__":
    main()
