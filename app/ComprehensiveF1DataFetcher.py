#!/usr/bin/env python3
"""
Comprehensive F1 Data Fetcher for Real F1 Data
Fetches data from past 3 years, filters active drivers, and creates training data
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class ComprehensiveF1DataFetcher:
    """Fetches comprehensive real F1 data from OpenF1 API"""
    
    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://api.openf1.org/v1"
        self.api_delay = 0.5  # Increased delay to avoid rate limiting
        
        # Define active drivers (2024 season) - these are the drivers we want to focus on
        self.active_drivers_2024 = {
            1: "Max Verstappen",      # Red Bull Racing
            11: "Sergio Perez",       # Red Bull Racing
            16: "Charles Leclerc",    # Ferrari
            55: "Carlos Sainz",       # Ferrari
            44: "Lewis Hamilton",     # Mercedes
            63: "George Russell",     # Mercedes
            4: "Lando Norris",        # McLaren
            81: "Oscar Piastri",      # McLaren
            14: "Fernando Alonso",    # Aston Martin
            18: "Lance Stroll",       # Aston Martin
            10: "Pierre Gasly",       # Alpine
            31: "Esteban Ocon",       # Alpine
            23: "Alexander Albon",    # Williams
            2: "Logan Sargeant",      # Williams
            20: "Kevin Magnussen",    # Haas
            27: "Nico Hulkenberg",    # Haas
            24: "Zhou Guanyu",        # Kick Sauber
            77: "Valtteri Bottas",    # Kick Sauber
            22: "Yuki Tsunoda",       # AlphaTauri
            3: "Daniel Ricciardo"     # AlphaTauri
        }
        
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[List[Dict]]:
        """Make API request with error handling and rate limiting"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            time.sleep(self.api_delay)  # Rate limiting
            return response.json()
        except Exception as e:
            logger.warning(f"API request failed for {endpoint}: {e}")
            return None
    
    def get_meetings(self, year: int) -> Optional[List[Dict]]:
        """Get F1 meetings (races) for a year"""
        return self._make_request("meetings", {"year": year})
    
    def get_sessions(self, meeting_key: int = None, session_type: str = "Race") -> Optional[List[Dict]]:
        """Get sessions for a meeting or all race sessions"""
        params = {}
        if meeting_key:
            params["meeting_key"] = meeting_key
        if session_type:
            params["session_type"] = session_type
        return self._make_request("sessions", params)
    
    def get_drivers(self) -> Optional[List[Dict]]:
        """Get all drivers"""
        return self._make_request("drivers")
    
    def get_positions(self, session_key: int) -> Optional[List[Dict]]:
        """Get position data for a session"""
        return self._make_request("position", {"session_key": session_key})
    
    def is_active_driver(self, driver_number: int, driver_name: str) -> bool:
        """Check if driver is active (in 2024 season)"""
        # Check if driver number is in active drivers list
        if driver_number in self.active_drivers_2024:
            return True
        
        # Check if driver name matches any active driver (case insensitive)
        active_names = [name.lower() for name in self.active_drivers_2024.values()]
        if driver_name.lower() in active_names:
            return True
            
        return False
    
    def get_race_results(self, session_key: int, session_info: Dict) -> Dict[str, Any]:
        """Get race results for a session, filtering for active drivers only"""
        try:
            # Get position data
            positions = self.get_positions(session_key)
            if not positions:
                logger.warning(f"No position data found for session {session_key}")
                return {}
            
            # Get driver data
            drivers = self.get_drivers()
            if not drivers:
                logger.warning("No driver data found")
                return {}
            
            # Create driver lookup
            driver_lookup = {d['driver_number']: d for d in drivers}
            
            # Get final positions for each driver
            driver_final_positions = {}
            for pos in positions:
                driver_num = pos['driver_number']
                if driver_num not in driver_final_positions:
                    driver_final_positions[driver_num] = []
                driver_final_positions[driver_num].append(pos['position'])
            
            # Create race results, filtering for active drivers only
            race_results = []
            for driver_num, position_list in driver_final_positions.items():
                if driver_num in driver_lookup:
                    driver = driver_lookup[driver_num]
                    driver_name = driver.get('full_name', f'Driver {driver_num}')
                    
                    # Only include active drivers
                    if self.is_active_driver(driver_num, driver_name):
                        final_position = position_list[-1] if position_list else None
                        
                        if final_position:
                            race_results.append({
                                'driver_number': driver_num,
                                'name': driver_name,
                                'team': driver.get('team_name', 'Unknown'),
                                'position': final_position,
                                'points_scored': self._calculate_points(final_position),
                                'fantasy_points': self.calculate_fantasy_points(
                                    final_position, 
                                    self._calculate_points(final_position)
                                ),
                                'session_key': session_key,
                                'meeting_key': session_info.get('meeting_key'),
                                'year': session_info.get('year'),
                                'location': session_info.get('location', 'Unknown')
                            })
            
            # Sort by position
            race_results.sort(key=lambda x: x['position'])
            return race_results
            
        except Exception as e:
            logger.error(f"Error getting race results: {e}")
            return {}
    
    def _calculate_points(self, position: int) -> float:
        """Calculate F1 points based on position"""
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        return points_system.get(position, 0)
    
    def calculate_fantasy_points(self, position: int, points_scored: float, 
                               fastest_lap: bool = False, pole_position: bool = False) -> float:
        """Calculate fantasy points based on F1 Fantasy rules"""
        # Position points (reverse order - 1st gets most points)
        position_points = max(0, 21 - position) * 2
        
        # Points scored bonus
        points_bonus = points_scored * 3
        
        # Fastest lap bonus
        fastest_lap_bonus = 5 if fastest_lap else 0
        
        # Pole position bonus
        pole_bonus = 3 if pole_position else 0
        
        total_fantasy_points = position_points + points_bonus + fastest_lap_bonus + pole_bonus
        return total_fantasy_points
    
    def get_comprehensive_f1_data(self, years: List[int] = [2022, 2023, 2024]) -> Dict[str, Any]:
        """Fetch comprehensive F1 data for multiple years"""
        logger.info(f"Fetching comprehensive F1 data for years: {years}")
        
        all_data = {
            'drivers': [],
            'meetings': [],
            'race_sessions': [],
            'race_results': {}
        }
        
        try:
            # Get drivers (once, as they don't change much)
            drivers = self.get_drivers()
            if drivers:
                # Filter for active drivers only
                active_drivers = []
                for driver in drivers:
                    driver_name = driver.get('full_name', f'Driver {driver.get("driver_number", "Unknown")}')
                    if self.is_active_driver(driver.get('driver_number'), driver_name):
                        active_drivers.append(driver)
                
                all_data['drivers'] = active_drivers
                logger.info(f"Found {len(active_drivers)} active drivers")
            
            # Get data for each year
            for year in years:
                logger.info(f"Processing year {year}...")
                
                # Get meetings for this year
                meetings = self.get_meetings(year)
                if meetings:
                    all_data['meetings'].extend(meetings)
                    logger.info(f"  Found {len(meetings)} meetings")
                
                # Get race sessions for this year
                race_sessions = self.get_sessions(session_type="Race")
                if race_sessions:
                    year_race_sessions = [s for s in race_sessions if s.get('year') == year]
                    all_data['race_sessions'].extend(year_race_sessions)
                    logger.info(f"  Found {len(year_race_sessions)} race sessions")
                    
                    # Get race results for each race (limit to avoid rate limiting)
                    max_races_per_year = 5  # Limit to avoid rate limiting
                    for i, session in enumerate(year_race_sessions[:max_races_per_year]):
                        session_key = session['session_key']
                        meeting_name = f"{session.get('location', 'Unknown')} {year}"
                        
                        logger.info(f"    Fetching results for {meeting_name}...")
                        race_results = self.get_race_results(session_key, session)
                        
                        if race_results:
                            all_data['race_results'][meeting_name] = race_results
                            logger.info(f"      Found {len(race_results)} active driver results")
                        else:
                            logger.warning(f"      No results found for {meeting_name}")
            
            logger.info(f"Successfully fetched data:")
            logger.info(f"  Active drivers: {len(all_data['drivers'])}")
            logger.info(f"  Meetings: {len(all_data['meetings'])}")
            logger.info(f"  Race sessions: {len(all_data['race_sessions'])}")
            logger.info(f"  Race results: {len(all_data['race_results'])}")
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive F1 data: {e}")
            return all_data
    
    def create_training_data(self, years: List[int] = [2022, 2023, 2024]) -> pd.DataFrame:
        """Create training data from real F1 data for multiple years"""
        try:
            # Get comprehensive data
            f1_data = self.get_comprehensive_f1_data(years)
            
            if not f1_data or not f1_data.get('race_results'):
                logger.error("No race results available")
                return pd.DataFrame()
            
            # Create training data
            training_data = []
            race_week = 1
            
            for race_name, results in f1_data['race_results'].items():
                for result in results:
                    # Generate realistic price based on performance and year
                    base_price = 20.0  # Base price in millions
                    performance_multiplier = 1.0 + (result['points_scored'] * 0.1)
                    
                    # Adjust price based on year (more recent = higher base price)
                    year_multiplier = 1.0 + (result['year'] - 2022) * 0.1
                    price = base_price * performance_multiplier * year_multiplier
                    
                    training_data.append({
                        'driver_number': result['driver_number'],
                        'name': result['name'],
                        'team': result['team'],
                        'price': round(price, 2),
                        'position': result['position'],
                        'points_scored': result['points_scored'],
                        'fantasy_points': result['fantasy_points'],
                        'season': result['year'],
                        'race_week': race_week,
                        'location': result['location']
                    })
                
                race_week += 1
            
            df = pd.DataFrame(training_data)
            logger.info(f"Created training data with {len(df)} records from {len(f1_data['race_results'])} races")
            
            # Log summary statistics
            logger.info(f"Training data summary:")
            logger.info(f"  Unique drivers: {df['name'].nunique()}")
            logger.info(f"  Unique teams: {df['team'].nunique()}")
            logger.info(f"  Years covered: {sorted(df['season'].unique())}")
            logger.info(f"  Fantasy points range: {df['fantasy_points'].min():.1f} - {df['fantasy_points'].max():.1f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            return pd.DataFrame()

def main():
    """Test the comprehensive F1 data fetcher"""
    print("Testing Comprehensive F1 Data Fetcher...")
    
    fetcher = ComprehensiveF1DataFetcher()
    
    # Test training data creation
    print("\nCreating training data from past 3 years...")
    training_data = fetcher.create_training_data([2022, 2023, 2024])
    
    if not training_data.empty:
        print(f"\n✅ Successfully created training data!")
        print(f"  Records: {len(training_data)}")
        print(f"  Drivers: {training_data['name'].nunique()}")
        print(f"  Teams: {training_data['team'].nunique()}")
        print(f"  Years: {sorted(training_data['season'].unique())}")
        print(f"  Races: {training_data['race_week'].nunique()}")
        
        print(f"\nSample data:")
        print(training_data[['name', 'team', 'position', 'fantasy_points', 'season']].head(10))
        
        # Show driver summary
        print(f"\nDriver performance summary:")
        driver_summary = training_data.groupby('name').agg({
            'fantasy_points': ['mean', 'max', 'count'],
            'position': 'mean'
        }).round(2)
        print(driver_summary.head(10))
        
    else:
        print("❌ Failed to create training data")

if __name__ == "__main__":
    main()
