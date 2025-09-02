import requests
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management for the application"""
    def __init__(self):
        self.fantasy_api_base = os.getenv('FANTASY_API_BASE', "https://fantasy-api.formula1.com/partner_games/f1")
        self.openf1_api_base = os.getenv('OPENF1_API_BASE', "https://api.openf1.org/v1")
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_name = os.getenv('DB_NAME', 'f1_data')
        self.db_user = os.getenv('DB_USER', 'root')
        self.db_password = os.getenv('DB_PASSWORD', '')
        self.fantasy_username = os.getenv('FANTASY_USERNAME')
        self.fantasy_password = os.getenv('FANTASY_PASSWORD')
        self.model_save_path = os.getenv('MODEL_SAVE_PATH', './models')
        self.data_save_path = os.getenv('DATA_SAVE_PATH', './data')

class F1DataFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.fantasy_api_base = config.fantasy_api_base
        self.openf1_api_base = config.openf1_api_base
        self.session = requests.Session()
        self._auth_token = None
        
    def authenticate_fantasy(self) -> bool:
        """Authenticate with Fantasy F1 API"""
        try:
            if not self.config.fantasy_username or not self.config.fantasy_password:
                logger.warning("Fantasy F1 credentials not provided. Using demo mode.")
                return False
                
            # This is a placeholder for the actual authentication flow
            # In practice, you would implement the specific auth mechanism for Fantasy F1
            auth_data = {
                'username': self.config.fantasy_username,
                'password': self.config.fantasy_password
            }
            
            # Example auth endpoint - adjust based on actual API
            auth_url = f"{self.fantasy_api_base}/auth/login"
            response = self.session.post(auth_url, json=auth_data)
            
            if response.status_code == 200:
                auth_response = response.json()
                self._auth_token = auth_response.get('token')
                logger.info("Successfully authenticated with Fantasy F1 API")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False
    
    def get_fantasy_cookie(self) -> Optional[str]:
        """Get authentication cookie for Fantasy F1 API"""
        if not self._auth_token:
            if not self.authenticate_fantasy():
                return None
        return self._auth_token
    
    def fetch_fantasy_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch data from Fantasy F1 API"""
        try:
            url = f"{self.fantasy_api_base}/{endpoint}"
            headers = {
                'Authorization': f'Bearer {self.get_fantasy_cookie()}',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching fantasy data from {endpoint}: {e}")
            return None
    
    def fetch_openf1_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch data from OpenF1 API"""
        try:
            url = f"{self.openf1_api_base}/{endpoint}"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching OpenF1 data from {endpoint}: {e}")
            return None
    
    def get_driver_standings(self, season: int = 2024) -> Optional[Dict[str, Any]]:
        """Get driver standings from Fantasy F1"""
        return self.fetch_fantasy_data("drivers", {"season": season})
    
    def get_driver_prices(self, season: int = 2024) -> Optional[Dict[str, Any]]:
        """Get driver prices from Fantasy F1"""
        return self.fetch_fantasy_data("driver_prices", {"season": season})
    
    def get_race_results(self, season: int = 2024) -> Optional[Dict[str, Any]]:
        """Get race results from Fantasy F1"""
        return self.fetch_fantasy_data("race_results", {"season": season})
    
    def get_car_telemetry(self, driver_number: int, session_key: str) -> Optional[Dict[str, Any]]:
        """Get car telemetry from OpenF1"""
        params = {
            'driver_number': driver_number,
            'session_key': session_key
        }
        return self.fetch_openf1_data("car_data", params)
    
    def get_session_info(self, year: int = 2024) -> Optional[Dict[str, Any]]:
        """Get session information from OpenF1"""
        return self.fetch_openf1_data("sessions", {"year": year})
    
    def get_drivers(self, season: int = 2024) -> Optional[Dict[str, Any]]:
        """Get driver information from OpenF1"""
        return self.fetch_openf1_data("drivers", {"year": season})

class MySQLDatabase:
    def __init__(self, config: Config):
        self.config = config
        self.host = config.db_host
        self.database = config.db_name
        self.user = config.db_user
        self.password = config.db_password
        self.connection = None
        
    def connect(self) -> bool:
        """Create database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            if self.connection.is_connected():
                logger.info("Connected to MySQL database")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if database connection is still valid"""
        try:
            if self.connection and self.connection.is_connected():
                # Test the connection with a simple query
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                return True
            return False
        except Error:
            return False
    
    def ensure_connection(self) -> bool:
        """Ensure database connection is active, reconnect if necessary"""
        if not self.is_connected():
            logger.info("Database connection lost, attempting to reconnect...")
            return self.connect()
        return True
    
    def create_tables(self) -> bool:
        """Create necessary tables"""
        if not self.ensure_connection():
            logger.error("Cannot create tables: no database connection")
            return False
            
        try:
            cursor = self.connection.cursor()
            
            # Drivers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drivers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    driver_number INT UNIQUE,
                    name VARCHAR(100),
                    team VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Driver prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS driver_prices (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    driver_number INT,
                    season INT,
                    race_week INT,
                    price DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (driver_number) REFERENCES drivers(driver_number)
                )
            """)
            
            # Race results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS race_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    driver_number INT,
                    season INT,
                    race_week INT,
                    position INT,
                    points_scored DECIMAL(10,2),
                    fantasy_points DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (driver_number) REFERENCES drivers(driver_number)
                )
            """)
            
            # Car telemetry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS car_telemetry (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    driver_number INT,
                    session_key VARCHAR(50),
                    date DATETIME,
                    speed DECIMAL(10,2),
                    throttle DECIMAL(5,2),
                    brake BOOLEAN,
                    drs BOOLEAN,
                    gear INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (driver_number) REFERENCES drivers(driver_number)
                )
            """)
            
            self.connection.commit()
            logger.info("Tables created successfully")
            return True
            
        except Error as e:
            logger.error(f"Error creating tables: {e}")
            return False
    
    def insert_driver(self, driver_number: int, name: str, team: str) -> bool:
        """Insert driver data"""
        if not self.ensure_connection():
            logger.error("Cannot insert driver: no database connection")
            return False
            
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT IGNORE INTO drivers (driver_number, name, team)
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (driver_number, name, team))
            self.connection.commit()
            logger.info(f"Driver {name} inserted/updated successfully")
            return True
        except Error as e:
            logger.error(f"Error inserting driver: {e}")
            return False
    
    def insert_race_result(self, driver_number: int, season: int, race_week: int, 
                          position: int, points_scored: float, fantasy_points: float) -> bool:
        """Insert race result data"""
        if not self.ensure_connection():
            logger.error("Cannot insert race result: no database connection")
            return False
            
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO race_results (driver_number, season, race_week, position, points_scored, fantasy_points)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (driver_number, season, race_week, position, points_scored, fantasy_points))
            self.connection.commit()
            logger.info(f"Race result for driver {driver_number} inserted successfully")
            return True
        except Error as e:
            logger.error(f"Error inserting race result: {e}")
            return False
    
    def insert_driver_price(self, driver_number: int, season: int, race_week: int, price: float) -> bool:
        """Insert driver price data"""
        if not self.ensure_connection():
            logger.error("Cannot insert driver price: no database connection")
            return False
            
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO driver_prices (driver_number, season, race_week, price)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (driver_number, season, race_week, price))
            self.connection.commit()
            logger.info(f"Driver price for driver {driver_number} inserted successfully")
            return True
        except Error as e:
            logger.error(f"Error inserting driver price: {e}")
            return False
    
    def get_training_data(self) -> Optional[pd.DataFrame]:
        """Get data for training the model"""
        if not self.ensure_connection():
            logger.error("Cannot fetch training data: no database connection")
            return None
            
        try:
            query = """
                SELECT 
                    d.driver_number,
                    d.name,
                    d.team,
                    dp.price,
                    rr.position,
                    rr.points_scored,
                    rr.fantasy_points,
                    rr.season,
                    rr.race_week
                FROM drivers d
                JOIN driver_prices dp ON d.driver_number = dp.driver_number
                JOIN race_results rr ON d.driver_number = rr.driver_number
                WHERE dp.season = rr.season AND dp.race_week = rr.race_week
                ORDER BY rr.season, rr.race_week
            """
            return pd.read_sql(query, self.connection)
        except Error as e:
            logger.error(f"Error fetching training data: {e}")
            return None

class F1Dataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CostEffectivenessModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32, 16], dropout_rate=0.2):
        super(CostEffectivenessModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))  # Output: cost-effectiveness score
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class F1ModelTrainer:
    def __init__(self, model, device='cpu', model_save_path='./models'):
        self.model = model
        self.device = device
        self.model_save_path = model_save_path
        self.model.to(device)
        
    def train_model(self, train_loader, val_loader, epochs=100, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for features, targets in train_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features)
                    val_loss += criterion(outputs.squeeze(), targets).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save to the configured path
                model_path = os.path.join(self.model_save_path, 'best_f1_model.pth')
                torch.save(self.model.state_dict(), model_path)
            
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            return self.model(features).cpu().numpy()

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the data for training"""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Validate required columns exist
    required_columns = ['fantasy_points', 'price', 'team']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Filter out rows with invalid data
    df = df.copy()
    df = df.dropna(subset=['fantasy_points', 'price', 'team'])
    
    # Filter out rows with zero or negative prices
    df = df[df['price'] > 0]
    
    if df.empty:
        raise ValueError("No valid data remaining after filtering")
    
    # Create cost-effectiveness target (points per dollar) with safety check
    df['cost_effectiveness'] = df.apply(
        lambda row: row['fantasy_points'] / row['price'] if row['price'] > 0 else 0, 
        axis=1
    )
    
    # Remove outliers (cost-effectiveness > 3 standard deviations from mean)
    mean_ce = df['cost_effectiveness'].mean()
    std_ce = df['cost_effectiveness'].std()
    df = df[abs(df['cost_effectiveness'] - mean_ce) <= 3 * std_ce]
    
    if df.empty:
        raise ValueError("No valid data remaining after outlier removal")
    
    # Encode categorical variables
    le_team = LabelEncoder()
    df['team_encoded'] = le_team.fit_transform(df['team'])
    
    # Select features
    feature_columns = ['driver_number', 'team_encoded', 'price', 'season', 'race_week']
    
    # Ensure all feature columns exist
    available_features = [col for col in feature_columns if col in df.columns]
    if len(available_features) < 3:  # Need at least 3 features
        raise ValueError(f"Insufficient features available: {available_features}")
    
    features = df[available_features].values
    targets = df['cost_effectiveness'].values
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    logger.info(f"Preprocessed data shape: {features_scaled.shape}")
    logger.info(f"Features used: {available_features}")
    
    return features_scaled, targets, scaler, le_team, available_features

def create_sample_data(db: MySQLDatabase, fetcher: F1DataFetcher) -> bool:
    """Create sample data for demonstration purposes"""
    try:
        logger.info("Creating sample data for demonstration...")
        
        # Sample driver data
        sample_drivers = [
            (44, "Lewis Hamilton", "Mercedes"),
            (1, "Max Verstappen", "Red Bull Racing"),
            (16, "Charles Leclerc", "Ferrari"),
            (4, "Lando Norris", "McLaren"),
            (63, "George Russell", "Mercedes"),
            (55, "Carlos Sainz", "Ferrari"),
            (11, "Sergio Perez", "Red Bull Racing"),
            (14, "Fernando Alonso", "Aston Martin"),
            (18, "Lance Stroll", "Aston Martin"),
            (81, "Oscar Piastri", "McLaren")
        ]
        
        # Insert drivers
        for driver_number, name, team in sample_drivers:
            if not db.insert_driver(driver_number, name, team):
                logger.error(f"Failed to insert driver {name}")
                return False
        
        # Sample race results and prices for multiple races
        sample_race_data = []
        for race_week in range(1, 6):  # 5 races
            for driver_number, _, _ in sample_drivers:
                # Generate realistic fantasy points (10-50 range)
                fantasy_points = np.random.uniform(10, 50)
                # Generate realistic position (1-20)
                position = np.random.randint(1, 21)
                # Generate realistic points scored (0-25)
                points_scored = max(0, 26 - position) if position <= 10 else 0
                # Generate realistic price (15-30 million)
                price = np.random.uniform(15, 30)
                
                sample_race_data.append({
                    'driver_number': driver_number,
                    'season': 2024,
                    'race_week': race_week,
                    'position': position,
                    'points_scored': round(points_scored, 2),
                    'fantasy_points': round(fantasy_points, 2),
                    'price': round(price, 2)
                })
        
        # Insert race results and prices
        for data in sample_race_data:
            if not db.insert_race_result(
                data['driver_number'], data['season'], data['race_week'],
                data['position'], data['points_scored'], data['fantasy_points']
            ):
                logger.error(f"Failed to insert race result for driver {data['driver_number']}")
                return False
            
            if not db.insert_driver_price(
                data['driver_number'], data['season'], data['race_week'], data['price']
            ):
                logger.error(f"Failed to insert driver price for driver {data['driver_number']}")
                return False
        
        logger.info(f"Successfully created sample data for {len(sample_drivers)} drivers across 5 races")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return False

def load_trained_model(config: Config, model_path: str = None):
    """Load a trained model and preprocessing objects"""
    try:
        if model_path is None:
            model_path = os.path.join(config.model_save_path, 'best_f1_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Reconstruct model
        model_config = checkpoint['model_config']
        model = CostEffectivenessModel(
            input_size=model_config['input_size'],
            hidden_sizes=model_config['hidden_sizes'],
            dropout_rate=model_config['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load preprocessing objects
        scaler_path = checkpoint.get('scaler_path', os.path.join(config.data_save_path, 'feature_scaler.pkl'))
        encoder_path = checkpoint.get('encoder_path', os.path.join(config.data_save_path, 'team_encoder.pkl'))
        
        import joblib
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        
        feature_names = checkpoint.get('feature_names', [])
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model, scaler, label_encoder, feature_names
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None, None

def predict_driver_performance(model, scaler, label_encoder, feature_names, 
                             driver_data: Dict[str, Any]) -> float:
    """Predict cost-effectiveness for a driver"""
    try:
        # Prepare features
        features = []
        for feature_name in feature_names:
            if feature_name == 'driver_number':
                features.append(driver_data.get('driver_number', 0))
            elif feature_name == 'team_encoded':
                team = driver_data.get('team', 'Unknown')
                features.append(label_encoder.transform([team])[0])
            elif feature_name == 'price':
                features.append(driver_data.get('price', 0.0))
            elif feature_name == 'season':
                features.append(driver_data.get('season', 2024))
            elif feature_name == 'race_week':
                features.append(driver_data.get('race_week', 1))
            else:
                features.append(0.0)  # Default value for unknown features
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(torch.FloatTensor(features_scaled))
            return prediction.item()
            
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return 0.0

def main():
    """Main execution function"""
    try:
        # Initialize configuration
        config = Config()
        
        # Create necessary directories
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.data_save_path, exist_ok=True)
        
        # Initialize components
        fetcher = F1DataFetcher(config)
        db = MySQLDatabase(config)
        
        # Connect to database and create tables
        if not db.connect():
            logger.error("Failed to connect to database. Exiting.")
            return
        
        if not db.create_tables():
            logger.error("Failed to create database tables. Exiting.")
            return
        
        # Try to fetch real data first
        logger.info("Attempting to fetch real F1 data...")
        real_data_available = False
        
        try:
            # Get session info from OpenF1
            sessions = fetcher.get_session_info(2024)
            if sessions:
                logger.info(f"Found {len(sessions)} sessions from OpenF1")
                real_data_available = True
            
            # Try to get driver data
            drivers = fetcher.get_drivers(2024)
            if drivers:
                logger.info(f"Found {len(drivers)} drivers from OpenF1")
                real_data_available = True
                
        except Exception as e:
            logger.warning(f"Could not fetch real data: {e}")
        
        # If no real data, create sample data
        if not real_data_available:
            logger.info("No real data available, creating sample data for demonstration...")
            if not create_sample_data(db, fetcher):
                logger.error("Failed to create sample data. Exiting.")
                return
        
        # Get training data
        training_data = db.get_training_data()
        
        if training_data is None or len(training_data) == 0:
            logger.error("No training data available. Exiting.")
            return
        
        logger.info(f"Training data shape: {training_data.shape}")
        logger.info(f"Columns: {list(training_data.columns)}")
        
        # Preprocess data
        try:
            features, targets, scaler, label_encoder, available_features = preprocess_data(training_data)
        except ValueError as e:
            logger.error(f"Data preprocessing failed: {e}")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Create datasets and loaders
        train_dataset = F1Dataset(X_train, y_train)
        val_dataset = F1Dataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize and train model
        model = CostEffectivenessModel(input_size=len(available_features))
        trainer = F1ModelTrainer(model, model_save_path=config.model_save_path)
        
        logger.info("Training model...")
        trainer.train_model(train_loader, val_loader, epochs=100)
        
        # Make predictions
        predictions = trainer.predict(X_test)
        
        # Evaluate model
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        logger.info(f"Test MSE: {mse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        
        # Save preprocessing objects
        try:
            import joblib
            scaler_path = os.path.join(config.data_save_path, 'feature_scaler.pkl')
            encoder_path = os.path.join(config.data_save_path, 'team_encoder.pkl')
            
            joblib.dump(scaler, scaler_path)
            joblib.dump(label_encoder, encoder_path)
            
            logger.info(f"Preprocessing objects saved to {config.data_save_path}")
        except Exception as e:
            logger.error(f"Failed to save preprocessing objects: {e}")
        
        # Save model
        try:
            model_path = os.path.join(config.model_save_path, 'best_f1_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': len(available_features),
                    'hidden_sizes': [64, 32, 16],
                    'dropout_rate': 0.2
                },
                'feature_names': available_features,
                'scaler_path': scaler_path,
                'encoder_path': encoder_path
            }, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise

if __name__ == "__main__":
    main()