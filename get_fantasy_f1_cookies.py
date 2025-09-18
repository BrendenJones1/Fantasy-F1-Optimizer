#!/usr/bin/env python3
"""
Script to get Fantasy F1 cookie data for authentication
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from FantasyF1API import FantasyF1API
import json

def get_cookie_data(email, password):
    """Get Fantasy F1 cookie data"""
    print("🔑 Getting Fantasy F1 Cookie Data...")
    print("=" * 50)
    
    # Initialize API client
    api = FantasyF1API()
    
    # Step 1: Get Reese84 cookie
    print("1. Getting Reese84 cookie (anti-bot detection)...")
    if api.get_reese84_cookie():
        print("✅ Reese84 cookie obtained successfully")
        print(f"   Token: {api.reese84_token[:50]}...")
        print(f"   Expires: {api.token_expires_at}")
    else:
        print("❌ Failed to get Reese84 cookie")
        return None
    
    # Step 2: Login with email/password
    print("\n2. Logging in with email/password...")
    if api.login_by_password(email, password):
        print("✅ Login successful")
        print(f"   Subscription Token: {api.subscription_token[:50]}...")
        print(f"   X-F1-Cookie-Data: {api.x_f1_cookie_data}")
        
        # Return the cookie data
        return {
            'reese84_token': api.reese84_token,
            'subscription_token': api.subscription_token,
            'x_f1_cookie_data': api.x_f1_cookie_data,
            'expires_at': api.token_expires_at.isoformat() if api.token_expires_at else None
        }
    else:
        print("❌ Login failed")
        return None

def main():
    """Main function"""
    print("Fantasy F1 Cookie Data Extractor")
    print("=" * 50)
    
    # Get credentials from user
    email = input("Enter your Fantasy F1 email: ").strip()
    password = input("Enter your Fantasy F1 password: ").strip()
    
    if not email or not password:
        print("❌ Email and password are required")
        return
    
    # Get cookie data
    cookie_data = get_cookie_data(email, password)
    
    if cookie_data:
        print("\n" + "=" * 50)
        print("🎉 SUCCESS! Your Fantasy F1 Cookie Data:")
        print("=" * 50)
        print(json.dumps(cookie_data, indent=2))
        
        print("\n" + "=" * 50)
        print("📝 How to use this data:")
        print("=" * 50)
        print("1. Save this data securely (it contains authentication tokens)")
        print("2. Use the 'x_f1_cookie_data' value in your API requests")
        print("3. The token expires, so you'll need to refresh it periodically")
        
        # Save to file
        with open('fantasy_f1_cookies.json', 'w') as f:
            json.dump(cookie_data, f, indent=2)
        print(f"\n💾 Cookie data saved to: fantasy_f1_cookies.json")
        
    else:
        print("\n❌ Failed to get cookie data")
        print("Check your email/password and try again")

if __name__ == "__main__":
    main()
