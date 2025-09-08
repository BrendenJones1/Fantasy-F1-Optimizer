# Fantasy F1 Optimizer

A simple machine learning system that predicts driver cost-effectiveness for Fantasy Formula 1 team selection.

## Quick Start

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Set Up MySQL Database

Create a database called `f1_data` in MySQL Workbench, or run:

```sql
CREATE DATABASE f1_data;
```

### 3. Update Database Credentials

Edit `app/DataPipelineAndModel.py` and change these lines in the `Config` class:

```python
self.db_user = 'root'  # your MySQL username
self.db_password = ''   # your MySQL password here
```

### 4. Train the Model

```bash
python3 app/DataPipelineAndModel.py
```

This will:
- Connect to your MySQL database
- Create the necessary tables
- Generate sample F1 data
- Train a neural network model
- Save the trained model

### 5. Make Predictions

```bash
python3 example_usage.py
```

This will:
- Load the trained model
- Make predictions for sample drivers
- Show cost-effectiveness rankings
- Provide team selection recommendations

## What It Does

- **Predicts driver cost-effectiveness** (fantasy points per dollar)
- **Uses neural networks** to analyze historical performance data
- **Generates sample data** for demonstration (no real F1 API needed)
- **Provides team optimization** recommendations based on budget

## Files

- `app/DataPipelineAndModel.py` - Main training script
- `example_usage.py` - Example of using the trained model
- `requirements.txt` - Python dependencies
- `database_setup.sql` - MySQL setup commands

## Database Schema

- **drivers** - Driver information (number, name, team)
- **driver_prices** - Historical pricing data
- **race_results** - Race performance including fantasy points

## Configuration

All settings are in the `Config` class in `app/DataPipelineAndModel.py`. Just change the database credentials to match your MySQL setup.

## Troubleshooting

- **Database connection error**: Check your MySQL credentials in the Config class
- **Import errors**: Make sure you're in the project root directory
- **Model not found**: Run the training script first

## No Virtual Environment Needed

This project works with your system Python installation. Just install the requirements and run!