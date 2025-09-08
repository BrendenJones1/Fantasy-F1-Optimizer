#!/usr/bin/env python3
"""
Simple run script for Fantasy F1 Optimizer
"""

import os
import sys

def main():
    print("Fantasy F1 Optimizer")
    print("=" * 30)
    print()
    print("Choose an option:")
    print("1. Train the model")
    print("2. Make predictions")
    print("3. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nTraining the model...")
            os.system("python3 app/DataPipelineAndModel.py")
            break
        elif choice == '2':
            print("\nMaking predictions...")
            os.system("python3 example_usage.py")
            break
        elif choice == '3':
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 