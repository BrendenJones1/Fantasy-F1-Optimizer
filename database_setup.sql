-- Fantasy F1 Optimizer Database Setup
-- Run this script to create the database and user

-- Create database
CREATE DATABASE IF NOT EXISTS f1_data;
USE f1_data;

-- Create user (adjust username and password as needed)
-- CREATE USER 'f1_user'@'localhost' IDENTIFIED BY 'your_password_here';
-- GRANT ALL PRIVILEGES ON f1_data.* TO 'f1_user'@'localhost';
-- FLUSH PRIVILEGES;

-- The tables will be created automatically by the Python application
-- This script just sets up the database structure

-- Optional: Create indexes for better performance
-- CREATE INDEX idx_driver_prices_season_race ON driver_prices(season, race_week);
-- CREATE INDEX idx_race_results_season_race ON race_results(season, race_week);
-- CREATE INDEX idx_drivers_team ON drivers(team);

-- Show current databases
SHOW DATABASES;

-- Use the f1_data database
USE f1_data;

-- Show tables (will be empty initially)
SHOW TABLES;