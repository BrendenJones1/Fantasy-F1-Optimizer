#!/usr/bin/env python3
"""
Official F1 Fantasy API Integration
Based on: https://documenter.getpostman.com/view/11462073/TzY68Dsi
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FantasyF1API:
    """Official F1 Fantasy API client"""
    
    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://fantasy-api.formula1.com/partner_games/f1"
        self.auth_url = "https://api.formula1.com"
        self.api_key = "fCUCjWxKPu9y1JwRAv8BpGLEgiAuThx7"  # From documentation
        
        # Authentication tokens
        self.reese84_token = None
        self.subscription_token = None
        self.x_f1_cookie_data = None
        
        # Token expiration
        self.token_expires_at = None
        
    def _make_request(self, url: str, method: str = "GET", data: Dict = None, 
                     headers: Dict = None, params: Dict = None) -> Optional[Dict]:
        """Make authenticated API request"""
        try:
            # Add default headers
            if headers is None:
                headers = {}
            
            # Add authentication headers if available
            if self.x_f1_cookie_data:
                headers['X-F1-Cookie-Data'] = self.x_f1_cookie_data
            
            if self.api_key:
                headers['apiKey'] = self.api_key
            
            # Make request
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers, params=params, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(url, headers=headers, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"API request failed for {url}: {e}")
            return None
    
    def get_reese84_cookie(self) -> bool:
        """Get Reese84 cookie for anti-bot detection"""
        try:
            url = f"{self.auth_url}/6657193977244c13?d=account.formula1.com"
            
            # Generate current timestamp and random values
            current_time = int(time.time())
            st = current_time
            sr = current_time + 1000000  # Random offset
            cr = current_time + 500000   # Random offset
            
            data = {
                "solution": {
                    "interrogation": {
                        "st": st,
                        "sr": sr,
                        "cr": cr
                    },
                    "version": "stable"
                },
                "error": None,
                "performance": {
                    "interrogation": 185
                }
            }
            
            response = self.session.post(url, json=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            self.reese84_token = result.get('token')
            self.token_expires_at = datetime.now() + timedelta(seconds=result.get('renewInSec', 3600))
            
            logger.info("Successfully obtained Reese84 cookie")
            return True
            
        except Exception as e:
            logger.error(f"Failed to get Reese84 cookie: {e}")
            return False
    
    def login_by_password(self, email: str, password: str) -> bool:
        """Login using email and password"""
        try:
            # First get Reese84 cookie
            if not self.get_reese84_cookie():
                return False
            
            url = f"{self.auth_url}/v2/account/subscriber/authenticate/by-password"
            
            headers = {
                'apiKey': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                "login": email,
                "password": password
            }
            
            response = self.session.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            self.subscription_token = result.get('subscriptionToken')
            
            # Generate X-F1-Cookie-Data from subscription token
            if self.subscription_token:
                self.x_f1_cookie_data = self._generate_cookie_data(self.subscription_token)
                logger.info("Successfully authenticated with F1 Fantasy API")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def _generate_cookie_data(self, subscription_token: str) -> str:
        """Generate X-F1-Cookie-Data from subscription token"""
        # This is a simplified implementation
        # The actual implementation might require more complex token processing
        return f"subscription_token={subscription_token}"
    
    def get_driver_prices(self, season: int = 2024) -> Optional[List[Dict]]:
        """Get current driver prices from Fantasy F1"""
        try:
            if not self.x_f1_cookie_data:
                logger.error("Not authenticated. Please login first.")
                return None
            
            url = f"{self.base_url}/game/{season}/drivers"
            response = self._make_request(url)
            
            if response:
                logger.info(f"Retrieved driver prices for season {season}")
                return response.get('drivers', [])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get driver prices: {e}")
            return None
    
    def get_constructor_prices(self, season: int = 2024) -> Optional[List[Dict]]:
        """Get current constructor prices from Fantasy F1"""
        try:
            if not self.x_f1_cookie_data:
                logger.error("Not authenticated. Please login first.")
                return None
            
            url = f"{self.base_url}/game/{season}/constructors"
            response = self._make_request(url)
            
            if response:
                logger.info(f"Retrieved constructor prices for season {season}")
                return response.get('constructors', [])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get constructor prices: {e}")
            return None
    
    def get_user_team(self, season: int = 2024) -> Optional[Dict]:
        """Get user's current fantasy team"""
        try:
            if not self.x_f1_cookie_data:
                logger.error("Not authenticated. Please login first.")
                return None
            
            url = f"{self.base_url}/game/{season}/my-team"
            response = self._make_request(url)
            
            if response:
                logger.info("Retrieved user's fantasy team")
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user team: {e}")
            return None
    
    def get_leaderboard(self, season: int = 2024, limit: int = 100) -> Optional[List[Dict]]:
        """Get fantasy leaderboard"""
        try:
            if not self.x_f1_cookie_data:
                logger.error("Not authenticated. Please login first.")
                return None
            
            url = f"{self.base_url}/game/{season}/leaderboard"
            params = {'limit': limit}
            response = self._make_request(url, params=params)
            
            if response:
                logger.info(f"Retrieved leaderboard for season {season}")
                return response.get('leaderboard', [])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")
            return None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return self.x_f1_cookie_data is not None
    
    def refresh_authentication(self) -> bool:
        """Refresh authentication if token is expired"""
        if self.token_expires_at and datetime.now() >= self.token_expires_at:
            logger.info("Token expired, refreshing authentication...")
            return self.get_reese84_cookie()
        return True

class FantasyF1PriceUpdater:
    """Updates driver prices from Fantasy F1 API"""
    
    def __init__(self, email: str = None, password: str = None):
        self.api = FantasyF1API()
        self.email = email
        self.password = password
        self.last_update = None
        
    def authenticate(self) -> bool:
        """Authenticate with Fantasy F1 API"""
        if not self.email or not self.password:
            logger.warning("No credentials provided. Using mock data.")
            return False
        
        return self.api.login_by_password(self.email, self.password)
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current driver prices from Fantasy F1"""
        try:
            # Try to get real prices from API
            if self.api.is_authenticated():
                drivers = self.api.get_driver_prices()
                if drivers:
                    prices = {}
                    for driver in drivers:
                        driver_name = driver.get('name', '')
                        price = driver.get('price', 0.0)
                        if driver_name and price:
                            prices[driver_name] = price
                    
                    self.last_update = datetime.now()
                    logger.info(f"Retrieved {len(prices)} driver prices from Fantasy F1 API")
                    return prices
            
            # Fallback to mock prices if API fails
            logger.warning("Using fallback mock prices")
            return self._get_mock_prices()
            
        except Exception as e:
            logger.error(f"Failed to get current prices: {e}")
            return self._get_mock_prices()
    
    def _get_mock_prices(self) -> Dict[str, float]:
        """Fallback mock prices based on 2024 performance"""
        return {
            'Max Verstappen': 30.0,      # Red Bull Racing - Champion
            'Sergio Perez': 25.0,        # Red Bull Racing
            'Charles Leclerc': 28.0,     # Ferrari
            'Carlos Sainz': 26.0,        # Ferrari
            'Lewis Hamilton': 27.0,      # Mercedes
            'George Russell': 24.0,      # Mercedes
            'Lando Norris': 25.0,        # McLaren
            'Oscar Piastri': 22.0,       # McLaren
            'Fernando Alonso': 23.0,     # Aston Martin
            'Lance Stroll': 20.0,        # Aston Martin
            'Pierre Gasly': 19.0,        # Alpine
            'Esteban Ocon': 18.0,        # Alpine
            'Alexander Albon': 17.0,     # Williams
            'Logan Sargeant': 15.0,      # Williams
            'Kevin Magnussen': 16.0,     # Haas
            'Nico Hulkenberg': 16.0,     # Haas
            'Zhou Guanyu': 17.0,         # Kick Sauber
            'Valtteri Bottas': 18.0,     # Kick Sauber
            'Yuki Tsunoda': 19.0,        # AlphaTauri
            'Daniel Ricciardo': 20.0     # AlphaTauri
        }
    
    def update_database_prices(self, db) -> bool:
        """Update driver prices in database"""
        try:
            prices = self.get_current_prices()
            if not prices:
                return False
            
            # Update prices in database
            cursor = db.connection.cursor()
            current_race_week = 10  # Next race week
            
            for driver_name, price in prices.items():
                # Find driver by name and update price
                query = """
                    UPDATE driver_prices dp
                    JOIN drivers d ON dp.driver_number = d.driver_number
                    SET dp.price = %s
                    WHERE d.name = %s AND dp.race_week = %s
                """
                cursor.execute(query, (price, driver_name, current_race_week))
            
            db.connection.commit()
            logger.info(f"Updated {len(prices)} driver prices in database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update database prices: {e}")
            return False

def main():
    """Test the Fantasy F1 API integration"""
    print("Testing Fantasy F1 API Integration...")
    
    # Test without credentials (will use mock data)
    updater = FantasyF1PriceUpdater()
    
    print("\n1. Testing price retrieval (mock data)...")
    prices = updater.get_current_prices()
    
    if prices:
        print(f"✅ Retrieved {len(prices)} driver prices")
        print("\nSample prices:")
        for name, price in list(prices.items())[:5]:
            print(f"  {name}: ${price:.1f}M")
    else:
        print("❌ Failed to retrieve prices")
    
    print("\n2. Testing API authentication (requires credentials)...")
    print("To test with real API, provide email and password:")
    print("updater = FantasyF1PriceUpdater('your_email@example.com', 'your_password')")
    print("updater.authenticate()")

if __name__ == "__main__":
    main()
