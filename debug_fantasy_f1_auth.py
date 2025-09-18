#!/usr/bin/env python3
"""
Debug script for Fantasy F1 authentication issues
"""

import requests
import json
import time
from datetime import datetime

def debug_fantasy_f1_auth():
    """Debug Fantasy F1 authentication step by step"""
    print("🔍 Debugging Fantasy F1 Authentication")
    print("=" * 50)
    
    # Step 1: Test Reese84 cookie endpoint
    print("1. Testing Reese84 cookie endpoint...")
    try:
        url = "https://api.formula1.com/6657193977244c13?d=account.formula1.com"
        
        # Generate current timestamp
        current_time = int(time.time())
        data = {
            "solution": {
                "interrogation": {
                    "st": current_time,
                    "sr": current_time + 1000000,
                    "cr": current_time + 500000
                },
                "version": "stable"
            },
            "error": None,
            "performance": {
                "interrogation": 185
            }
        }
        
        response = requests.post(url, json=data, timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            result = response.json()
            reese84_token = result.get('token')
            print(f"   ✅ Reese84 token obtained: {reese84_token[:50]}...")
            
            # Step 2: Test login endpoint
            print("\n2. Testing login endpoint...")
            login_url = "https://api.formula1.com/v2/account/subscriber/authenticate/by-password"
            
            headers = {
                'apiKey': 'fCUCjWxKPu9y1JwRAv8BpGLEgiAuThx7',
                'Content-Type': 'application/json'
            }
            
            # Test with dummy credentials first
            login_data = {
                "login": "test@example.com",
                "password": "testpassword"
            }
            
            login_response = requests.post(login_url, headers=headers, json=login_data, timeout=10)
            print(f"   Status Code: {login_response.status_code}")
            print(f"   Response: {login_response.text[:200]}...")
            
            if login_response.status_code == 401:
                print("   ⚠️  401 Unauthorized - This is expected with dummy credentials")
                print("   This means the endpoint is working, but credentials are invalid")
            elif login_response.status_code == 200:
                print("   ✅ Login endpoint is working")
            else:
                print(f"   ❌ Unexpected status code: {login_response.status_code}")
                
        else:
            print(f"   ❌ Failed to get Reese84 token: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🔧 Troubleshooting Tips:")
    print("=" * 50)
    print("1. Make sure you have a valid Fantasy F1 account")
    print("2. Check if your account is active and not suspended")
    print("3. Try logging into fantasy.formula1.com manually first")
    print("4. The API might require a paid subscription")
    print("5. Check if there are any rate limiting issues")
    
    print("\n📝 Alternative Approach:")
    print("=" * 50)
    print("If authentication fails, you can still use the system with mock prices:")
    print("python3 example_usage_with_fantasy_api.py")
    print("This will use realistic mock prices based on 2024 performance.")

if __name__ == "__main__":
    debug_fantasy_f1_auth()
