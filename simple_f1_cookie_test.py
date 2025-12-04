#!/usr/bin/env python3
"""
Simple test of the exact Fantasy F1 authentication flow
"""

import requests
import json
import time

def test_f1_auth_flow():
    """Test the exact authentication flow from the documentation"""
    print("Testing Fantasy F1 Authentication Flow")
    print("=" * 50)
    
    # Step 1: Get Reese84 cookie
    print("1. Getting Reese84 cookie...")
    reese84_url = "https://api.formula1.com/6657193977244c13?d=account.formula1.com"
    
    current_time = int(time.time())
    reese84_data = {
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
    
    try:
        reese84_response = requests.post(reese84_url, json=reese84_data, timeout=10)
        print(f"   Status: {reese84_response.status_code}")
        
        if reese84_response.status_code == 200:
            reese84_result = reese84_response.json()
            reese84_token = reese84_result.get('token')
            print(f"   ✅ Reese84 token: {reese84_token[:50]}...")
        else:
            print(f"   ❌ Failed: {reese84_response.text}")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Step 2: Login with password (try different approaches)
    print("\n2. Testing login endpoint...")
    login_url = "https://api.formula1.com/v2/account/subscriber/authenticate/by-password"
    
    # Test data (you'll need to replace with real credentials)
    login_data = {
        "login": "your_email@example.com",
        "password": "your_password"
    }
    
    # Try different header combinations
    header_combinations = [
        {
            'name': 'With API Key + Cookie',
            'headers': {
                'apiKey': 'fCUCjWxKPu9y1JwRAv8BpGLEgiAuThx7',
                'Content-Type': 'application/json',
                'Cookie': f'reese84={reese84_token}'
            }
        },
        {
            'name': 'With API Key + X-Reese84',
            'headers': {
                'apiKey': 'fCUCjWxKPu9y1JwRAv8BpGLEgiAuThx7',
                'Content-Type': 'application/json',
                'X-Reese84-Token': reese84_token
            }
        },
        {
            'name': 'Without API Key + Cookie',
            'headers': {
                'Content-Type': 'application/json',
                'Cookie': f'reese84={reese84_token}'
            }
        },
        {
            'name': 'Without API Key + X-Reese84',
            'headers': {
                'Content-Type': 'application/json',
                'X-Reese84-Token': reese84_token
            }
        }
    ]
    
    for combo in header_combinations:
        print(f"\n   Testing: {combo['name']}")
        try:
            response = requests.post(login_url, headers=combo['headers'], json=login_data, timeout=10)
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:150]}...")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS! This combination works!")
                result = response.json()
                subscription_token = result.get('subscriptionToken')
                print(f"   Subscription Token: {subscription_token[:50]}...")
                
                # Step 3: Generate X-F1-Cookie-Data
                print(f"\n3. Generating X-F1-Cookie-Data...")
                print(f"   Subscription Token: {subscription_token}")
                print(f"   Note: The exact transformation method isn't documented.")
                print(f"   You may need to reverse-engineer this from browser requests.")
                
                return subscription_token
            elif response.status_code == 401:
                print(f"   ⚠️  Unauthorized (expected with test credentials)")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n❌ All authentication attempts failed.")
    print(f"This suggests the API key might be expired or the method has changed.")

if __name__ == "__main__":
    test_f1_auth_flow()
