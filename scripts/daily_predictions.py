#!/usr/bin/env python3
"""
Daily NBA prediction script
Run this daily to get predictions for today's games
"""
import subprocess
import sys
import os

def main():
    print("="*70)
    print("DAILY NBA PREDICTIONS")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    predict_script = os.path.join(script_dir, 'predict_upcoming_games.py')
    
    # Run prediction script
    result = subprocess.run([sys.executable, predict_script], capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Daily predictions generated successfully!")
    else:
        print("\n❌ Error generating predictions")

if __name__ == '__main__':
    main()

