# Fantasy F1 Optimizer

A machine learning system that optimizes Fantasy Formula 1 team selections by predicting driver cost-effectiveness scores. The system analyzes historical performance data, pricing, and race results to provide data-driven recommendations for optimal team composition.

## Features

- **Data Pipeline**: Integrates with Fantasy F1 and OpenF1 APIs to collect comprehensive driver data
- **Machine Learning Model**: Neural network-based prediction of driver cost-effectiveness (points per dollar)
- **Database Storage**: MySQL database with normalized schema for efficient data management
- **Sample Data Generation**: Built-in sample data creation for demonstration and testing
- **Model Persistence**: Save and load trained models for production use
- **Configuration Management**: Environment-based configuration for flexible deployment

## Architecture

The system consists of several key components:

1. **F1DataFetcher**: Handles API communication with Fantasy F1 and OpenF1
2. **MySQLDatabase**: Manages data storage and retrieval with proper connection handling
3. **CostEffectivenessModel**: PyTorch neural network for predicting driver performance
4. **F1ModelTrainer**: Handles model training with validation and early stopping
5. **Data Preprocessing**: Feature engineering, scaling, and categorical encoding

## Installation

### Prerequisites

- Python 3.8+
- MySQL 8.0+
- PyTorch (CPU or GPU version)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Fantasy-F1-Optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
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
```

4. Create the MySQL database:
```sql
CREATE DATABASE f1_data;
```

## Usage

### Training the Model

Run the main training script:

```bash
python app/DataPipelineAndModel
```

This will:
- Connect to the database and create necessary tables
- Attempt to fetch real F1 data (or create sample data if none available)
- Preprocess the data and train the neural network
- Save the trained model and preprocessing objects

### Using the Trained Model

```python
from app.DataPipelineAndModel import load_trained_model, predict_driver_performance
from config import Config

# Load configuration and model
config = Config()
model, scaler, label_encoder, feature_names = load_trained_model(config)

# Predict performance for a driver
driver_data = {
    'driver_number': 44,
    'team': 'Mercedes',
    'price': 25.5,
    'season': 2024,
    'race_week': 6
}

predicted_ce = predict_driver_performance(model, scaler, label_encoder, feature_names, driver_data)
print(f"Predicted cost-effectiveness: {predicted_ce:.4f}")
```

## Database Schema

The system uses a normalized database design with the following tables:

- **drivers**: Driver information (number, name, team)
- **driver_prices**: Historical pricing data by season and race week
- **race_results**: Race performance data including fantasy points
- **car_telemetry**: Detailed car performance metrics (future enhancement)

## Model Architecture

The neural network uses:
- **Input Layer**: Driver features (number, team, price, season, race week)
- **Hidden Layers**: 64 → 32 → 16 neurons with ReLU activation
- **Output Layer**: Single neuron predicting cost-effectiveness score
- **Regularization**: Dropout (20%) to prevent overfitting
- **Optimization**: Adam optimizer with learning rate scheduling

## Configuration

All configuration is managed through environment variables or the `Config` class. Key settings include:

- Database connection parameters
- API endpoints and authentication
- Model hyperparameters
- File paths for model and data storage

## Development

### Project Structure

```
Fantasy-F1-Optimizer/
├── app/
│   ├── DataCollectionAndPrediction
│   └── DataPipelineAndModel
├── config.py
├── database_setup.sql
├── requirements.txt
└── README.md
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
flake8 .
```

## Future Enhancements

- Real-time data updates during race weekends
- Team composition optimization algorithms
- Web interface for easy team management
- Integration with popular fantasy platforms
- Advanced analytics and visualization
- Driver form and momentum analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and entertainment purposes. Fantasy sports involve risk and no guarantee of profit. Always make informed decisions and never bet more than you can afford to lose.