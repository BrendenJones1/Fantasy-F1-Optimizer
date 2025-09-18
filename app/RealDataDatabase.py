#!/usr/bin/env python3
"""
Database operations for real F1 data
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class RealDataDatabase:
    """Database operations for real F1 data"""
    
    def __init__(self, config):
        self.config = config
        self.connection = None
        
    def connect(self) -> bool:
        """Create database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config.db_host,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            if self.connection.is_connected():
                logger.info(f"Connected to MySQL database: {self.config.db_name}")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
    
    def clear_sample_data(self) -> bool:
        """Clear existing sample data"""
        try:
            cursor = self.connection.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM race_results")
            cursor.execute("DELETE FROM driver_prices")
            cursor.execute("DELETE FROM drivers")
            
            self.connection.commit()
            logger.info("Cleared existing sample data")
            return True
            
        except Error as e:
            logger.error(f"Error clearing sample data: {e}")
            return False
    
    def insert_real_drivers(self, drivers_data: list) -> bool:
        """Insert real driver data"""
        try:
            cursor = self.connection.cursor()
            
            for driver in drivers_data:
                query = """
                    INSERT IGNORE INTO drivers (driver_number, name, team)
                    VALUES (%s, %s, %s)
                """
                cursor.execute(query, (
                    driver['driver_number'],
                    driver['name'],
                    driver['team']
                ))
            
            self.connection.commit()
            logger.info(f"Inserted {len(drivers_data)} real drivers")
            return True
            
        except Error as e:
            logger.error(f"Error inserting real drivers: {e}")
            return False
    
    def insert_real_race_data(self, training_data: pd.DataFrame) -> bool:
        """Insert real race data (results and prices)"""
        try:
            cursor = self.connection.cursor()
            
            for _, row in training_data.iterrows():
                # Insert race result
                race_query = """
                    INSERT INTO race_results (driver_number, season, race_week, position, points_scored, fantasy_points)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(race_query, (
                    row['driver_number'],
                    row['season'],
                    row['race_week'],
                    row['position'],
                    row['points_scored'],
                    row['fantasy_points']
                ))
                
                # Insert driver price
                price_query = """
                    INSERT INTO driver_prices (driver_number, season, race_week, price)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(price_query, (
                    row['driver_number'],
                    row['season'],
                    row['race_week'],
                    row['price']
                ))
            
            self.connection.commit()
            logger.info(f"Inserted {len(training_data)} real race records")
            return True
            
        except Error as e:
            logger.error(f"Error inserting real race data: {e}")
            return False
    
    def get_training_data(self) -> Optional[pd.DataFrame]:
        """Get data for training the model"""
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
    
    def get_active_drivers(self) -> Optional[pd.DataFrame]:
        """Get list of active drivers"""
        try:
            query = """
                SELECT DISTINCT driver_number, name, team
                FROM drivers
                ORDER BY name
            """
            return pd.read_sql(query, self.connection)
        except Error as e:
            logger.error(f"Error fetching active drivers: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")
