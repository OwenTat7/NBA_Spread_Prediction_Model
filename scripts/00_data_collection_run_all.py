#!/usr/bin/env python3
"""
Master script to run full data collection pipeline:
1. Fetch historical games with spreads
2. Add injury features
3. Ready for model training
"""
import subprocess
import sys
import os
from datetime import datetime

print("="*70)
print("NBA DATA COLLECTION - FULL PIPELINE")
print("="*70)
print("\nThis will:")
print("  1. Fetch historical games from ESPN API (with spreads)")
print("  2. Add injury features to games")
print("  3. Prepare data for model training")
print("\n⚠️  This will take 30-60 minutes due to API rate limiting")
print("="*70)

# Support non-interactive mode
non_interactive = '--non-interactive' in sys.argv or not sys.stdin.isatty()
if non_interactive:
    print("\nRunning in non-interactive mode...")
    response = 'y'
else:
    response = input("\nContinue? (y/n): ")

if response.lower() != 'y':
    print("Cancelled.")
    sys.exit(0)

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

# Step 1: Fetch historical games with spreads
print("\n" + "="*70)
print("STEP 1: Fetching Historical Games with Spreads")
print("="*70)
print("This will take ~20-40 minutes...\n")

try:
    result = subprocess.run(
        [sys.executable, os.path.join(script_dir, '01_data_collection_fetch_historical_games.py')],
        cwd=root_dir,
        check=True
    )
    print("\n✓ Step 1 complete: Historical games fetched")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Step 1 failed: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    sys.exit(1)

# Step 2: Add injury features
print("\n" + "="*70)
print("STEP 2: Adding Injury Features")
print("="*70)
print("This will take ~15-30 minutes...\n")

try:
    result = subprocess.run(
        [sys.executable, os.path.join(script_dir, '02_data_collection_add_injuries.py')],
        cwd=root_dir,
        check=True
    )
    print("\n✓ Step 2 complete: Injury features added")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Step 2 failed: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    sys.exit(1)

# Final summary
print("\n" + "="*70)
print("✅ DATA COLLECTION COMPLETE!")
print("="*70)
print("\nFiles created:")
print("  - data/final_dataset_raw_games.csv (games with spreads)")
print("  - data/final_dataset_with_injuries.csv (FINAL DATASET - ready for training)")

# Check if spreads were captured
try:
    import pandas as pd
    data_file = os.path.join(root_dir, 'data', 'final_dataset_with_injuries.csv')
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        if 'opening_spread' in df.columns:
            spread_count = df['opening_spread'].notna().sum()
            print(f"\n📊 Spread Data:")
            print(f"   Games with spreads: {spread_count}/{len(df)} ({spread_count/len(df)*100:.1f}%)")
        else:
            print("\n⚠️  Warning: No 'opening_spread' column found")
except:
    pass

print("\n✓ Ready for model training!")
print("="*70)

