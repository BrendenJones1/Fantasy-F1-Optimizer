#!/usr/bin/env python3
"""
Proper Fantasy F1 Cookie Extraction following the official API flow
Based on: https://documenter.getpostman.com/view/11462073/TzY68Dsi
"""

import requests
import json
import time
from datetime import datetime

def get_reese84_cookie():
    """Step 1: Get Reese84 cookie for anti-bot detection"""
    print("🔑 Step 1: Getting Reese84 cookie...")
    
    url = "https://api.formula1.com/6657193977244c13?d=account.formula1.com"
    
    # Generate current timestamp and random values
    current_time = int(time.time())
    data = {
        "solution": {
            "interrogation": {
                "st": current_time,
                "sr": current_time + 1000000,  # Random offset
                "cr": current_time + 500000    # Random offset
            },
            "version": "stable"
        },
        "error": None,
        "performance": {
            "interrogation": 185
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            token = result.get('token')
            renew_in_sec = result.get('renewInSec', 3600)
            cookie_domain = result.get('cookieDomain')
            
            print(f"   ✅ Reese84 token obtained")
            print(f"   Token: {token[:50]}...")
            print(f"   Renew in: {renew_in_sec} seconds")
            print(f"   Domain: {cookie_domain}")
            
            return token
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def login_with_password(email, password, reese84_token):
    """Step 2: Login with email/password using Reese84 token"""
    print("\n🔑 Step 2: Logging in with email/password...")
    
    url = "https://api.formula1.com/v2/account/subscriber/authenticate/by-password"
    
    # Try different header combinations
    header_options = [
        # Option 1: With API key (from documentation)
        {
            'apiKey': 'fCUCjWxKPu9y1JwRAv8BpGLEgiAuThx7',
            'Content-Type': 'application/json',
            'Cookie': f'reese84={reese84_token}'
        },
        # Option 2: Without API key
        {
            'Content-Type': 'application/json',
            'Cookie': f'reese84={reese84_token}'
        },
        # Option 3: With different header name
        {
            'Content-Type': 'application/json',
            'X-Reese84-Token': reese84_token
        },
        # Option 4: With Authorization header
        {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {reese84_token}'
        }
    ]
    
    login_data = {
        "login": email,
        "password": password
    }
    
    for i, headers in enumerate(header_options, 1):
        print(f"   Trying option {i}...")
        try:
            response = requests.post(url, headers=headers, json=login_data, timeout=10)
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                result = response.json()
                subscription_token = result.get('subscriptionToken')
                user_info = result.get('user', {})
                
                print(f"   ✅ Login successful!")
                print(f"   Subscription Token: {subscription_token[:50]}...")
                print(f"   User ID: {user_info.get('id', 'N/A')}")
                print(f"   Email: {user_info.get('email', 'N/A')}")
                
                return subscription_token
            elif response.status_code == 401:
                print(f"   ⚠️  Unauthorized (expected with invalid credentials)")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return None

def generate_x_f1_cookie_data(subscription_token):
    """Step 3: Generate X-F1-Cookie-Data from subscription token"""
    print("\n🔑 Step 3: Generating X-F1-Cookie-Data...")
    
    # This is where we need to implement the token transformation
    # The exact method isn't documented, but we can try different approaches
    
    print("   Note: The exact method to generate X-F1-Cookie-Data isn't documented.")
    print("   We'll try different approaches...")
    
    # Approach 1: Direct token
    x_f1_cookie_data = subscription_token
    print(f"   Approach 1 (direct): {x_f1_cookie_data[:50]}...")
    
    # Approach 2: Base64 encoded
    import base64
    try:
        encoded_token = base64.b64encode(subscription_token.encode()).decode()
        print(f"   Approach 2 (base64): {encoded_token[:50]}...")
    except:
        print("   Approach 2: Failed to encode")
    
    # Approach 3: JWT-like format
    try:
        # Split JWT token if it's in that format
        if '.' in subscription_token:
            parts = subscription_token.split('.')
            print(f"   Approach 3 (JWT parts): {len(parts)} parts")
            for i, part in enumerate(parts):
                print(f"     Part {i+1}: {part[:30]}...")
    except:
        print("   Approach 3: Not a JWT token")
    
    return x_f1_cookie_data

def test_fantasy_api_access(x_f1_cookie_data):
    """Step 4: Test access to Fantasy F1 API"""
    print("\n🔑 Step 4: Testing Fantasy F1 API access...")
    
    # Test different header formats
    header_options = [
        {'X-F1-Cookie-Data': x_f1_cookie_data},
        {'Authorization': f'Bearer {x_f1_cookie_data}'},
        {'Cookie': f'f1-token={x_f1_cookie_data}'},
        {'X-Auth-Token': x_f1_cookie_data}
    ]
    
    test_url = "https://fantasy-api.formula1.com/partner_games/f1/game/2024/drivers"
    
    for i, headers in enumerate(header_options, 1):
        print(f"   Testing option {i}...")
        try:
            response = requests.get(test_url, headers=headers, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ Success! API access granted")
                result = response.json()
                print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
                return True
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Response: {response.text[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return False

def main():
    """Main function to get Fantasy F1 cookie data"""
    print("Fantasy F1 Cookie Data Extractor")
    print("Following official API flow from documentation")
    print("=" * 60)
    
    # Get credentials
    email = input("Enter your Fantasy F1 email: ").strip()
    password = input("Enter your Fantasy F1 password: ").strip()
    
    if not email or not password:
        print("❌ Email and password are required")
        return
    
    # Step 1: Get Reese84 cookie
    reese84_token = get_reese84_cookie()
    if not reese84_token:
        print("❌ Failed to get Reese84 cookie")
        return
    
    # Step 2: Login with password
    subscription_token = login_with_password(email, password, reese84_token)
    if not subscription_token:
        print("❌ Failed to login")
        return
    
    # Step 3: Generate X-F1-Cookie-Data
    x_f1_cookie_data = generate_x_f1_cookie_data(subscription_token)
    
    # Step 4: Test API access
    api_access = test_fantasy_api_access(x_f1_cookie_data)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 AUTHENTICATION SUMMARY")
    print("=" * 60)
    print(f"Reese84 Token: {reese84_token[:50]}...")
    print(f"Subscription Token: {subscription_token[:50]}...")
    print(f"X-F1-Cookie-Data: {x_f1_cookie_data[:50]}...")
    print(f"API Access: {'✅ Success' if api_access else '❌ Failed'}")
    
    if api_access:
        print("\n💾 Save this data securely:")
        cookie_data = {
            'reese84_token': reese84_token,
            'subscription_token': subscription_token,
            'x_f1_cookie_data': x_f1_cookie_data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('fantasy_f1_cookies.json', 'w') as f:
            json.dump(cookie_data, f, indent=2)
        print("   Saved to: fantasy_f1_cookies.json")
    else:
        print("\n⚠️  API access failed. The X-F1-Cookie-Data generation method")
        print("   might need to be reverse-engineered from the browser.")

if __name__ == "__main__":
    main()
