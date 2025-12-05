#!/usr/bin/env python3
"""
Add injury features to existing NBA games dataset
Includes multicollinearity analysis and feature importance testing
"""
import pandas as pd
import numpy as np
from injury_features import (
    add_injury_features_to_games,
    analyze_injury_correlations,
    check_multicollinearity
)
import sys
import os

print("="*70)
print("INJURY FEATURES - ADDITION TO PIPELINE")
print("="*70)

# Load existing games data
print("\n[1/4] Loading games data...")
# Try new naming first, then fallback to old naming for compatibility
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
for filename in [
    os.path.join(data_dir, 'final_dataset_raw_games.csv'),
    os.path.join(data_dir, 'nba_games.csv'),
    'final_dataset_raw_games.csv',
    'nba_games.csv',
    'nba_games_test.csv'
]:
    try:
        games_df = pd.read_csv(filename, parse_dates=['date'])
        print(f"✓ Loaded {len(games_df)} games from {filename}")
        break
    except FileNotFoundError:
        continue
else:
    print("❌ No game data file found!")
    print(f"   Looking for: {os.path.join(data_dir, 'final_dataset_raw_games.csv')} or data/nba_games.csv")
    sys.exit(1)

# Check if injury features already exist
injury_cols = [c for c in games_df.columns if 'injury' in c.lower() or 'players_out' in c.lower()]
if injury_cols:
    print(f"\n⚠️  Injury features already exist: {injury_cols}")
    response = input("Re-fetch injury data? (y/n): ").lower()
    if response != 'y':
        print("Using existing injury features...")
        games_with_injuries = games_df
    else:
        # Remove existing injury columns
        games_df = games_df.drop(columns=injury_cols)
        games_with_injuries = add_injury_features_to_games(games_df)
        # Use final_dataset naming
        if 'final_dataset_raw_games.csv' in filename:
            output_file = filename.replace('final_dataset_raw_games.csv', 'final_dataset_with_injuries.csv')
        elif 'nba_games.csv' in filename:
            output_file = os.path.join(data_dir, 'final_dataset_with_injuries.csv')
        else:
            output_file = filename.replace('.csv', '_with_injuries.csv')
        games_with_injuries.to_csv(output_file, index=False)
        print(f"✓ Saved to {output_file}")
else:
    # Add injury features
    print("\n[2/4] Fetching injury data from ESPN API...")
    print("⚠️  This will take 15-30 minutes for full dataset (API rate limiting)")
    
    # Check if running non-interactively (background mode)
    import sys
    non_interactive = not sys.stdin.isatty() or '--non-interactive' in sys.argv
    
    if non_interactive:
        print("  Running in non-interactive mode (background)...")
        response = 'y'
    else:
        response = input("Continue? (y/n): ").lower()
    
    if response == 'y':
        # In non-interactive mode, process full dataset directly
        if non_interactive:
            print(f"\n  Processing full dataset ({len(games_df)} games)...")
            print(f"  This will take approximately {len(games_df) * 0.5 / 60:.1f} minutes")
            games_with_injuries = add_injury_features_to_games(games_df, batch_size=20, delay=0.3)
            # Use final_dataset naming
            if 'final_dataset_raw_games.csv' in filename:
                output_file = filename.replace('final_dataset_raw_games.csv', 'final_dataset_with_injuries.csv')
            elif 'nba_games.csv' in filename:
                output_file = os.path.join(data_dir, 'final_dataset_with_injuries.csv')
            else:
                output_file = filename.replace('.csv', '_with_injuries.csv')
            games_with_injuries.to_csv(output_file, index=False)
            print(f"\n✓ Saved to {output_file}")
        else:
            # Interactive mode: sample first 50 games for testing
            if len(games_df) > 50:
                print(f"\n⚠️  Large dataset ({len(games_df)} games). Testing with first 50 games...")
                test_df = games_df.head(50).copy()
                test_with_injuries = add_injury_features_to_games(test_df, batch_size=5, delay=0.3)
                
                print("\n✓ Test complete. Sample results:")
                print(test_with_injuries[['game_id', 'home_team_name', 'players_out_home', 
                                         'players_out_away', 'injury_severity_diff']].head(10))
                
                response = input("\nProcess full dataset? (y/n): ").lower()
                if response == 'y':
                    games_with_injuries = add_injury_features_to_games(games_df)
                    # Use final_dataset naming
                    if 'final_dataset_raw_games.csv' in filename:
                        output_file = filename.replace('final_dataset_raw_games.csv', 'final_dataset_with_injuries.csv')
                    elif 'nba_games.csv' in filename:
                        output_file = os.path.join(data_dir, 'final_dataset_with_injuries.csv')
                    else:
                        output_file = filename.replace('.csv', '_with_injuries.csv')
                    games_with_injuries.to_csv(output_file, index=False)
                    print(f"✓ Saved to {output_file}")
                else:
                    games_with_injuries = test_with_injuries
            else:
                games_with_injuries = add_injury_features_to_games(games_df)
                # Use final_dataset naming
                if 'final_dataset_raw_games.csv' in filename:
                    output_file = filename.replace('final_dataset_raw_games.csv', 'final_dataset_with_injuries.csv')
                elif 'nba_games.csv' in filename:
                    output_file = os.path.join(data_dir, 'final_dataset_with_injuries.csv')
                else:
                    output_file = filename.replace('.csv', '_with_injuries.csv')
                games_with_injuries.to_csv(output_file, index=False)
                print(f"✓ Saved to {output_file}")
    else:
        print("Skipping injury data fetch. Using existing data.")
        games_with_injuries = games_df

# Analyze correlations
print("\n[3/4] Analyzing multicollinearity...")
multicollinearity_results = check_multicollinearity(games_with_injuries, threshold=0.8)

if multicollinearity_results['warnings']:
    print("\n⚠️  MULTICOLLINEARITY WARNINGS:")
    for warning in multicollinearity_results['warnings']:
        print(f"  {warning}")
else:
    print("✓ No high correlations detected (threshold: 0.8)")

# Show correlation matrix
corr_matrix = analyze_injury_correlations(games_with_injuries)
if not corr_matrix.empty:
    print("\nCorrelation Matrix (Injury Features vs Spread Features):")
    print("="*70)
    print(corr_matrix.round(3))
    print("\nInterpretation:")
    print("  - Values close to 1.0 or -1.0 indicate high correlation")
    print("  - Values between -0.3 and 0.3 indicate low correlation")
    print("  - Moderate correlation (0.4-0.7) is acceptable and may add value")

# Summary statistics
print("\n[4/4] Injury Feature Summary:")
print("="*70)
injury_cols = [c for c in games_with_injuries.columns if any(x in c for x in [
    'injury', 'players_out', 'star_out', 'players_dtd'
])]

if injury_cols:
    summary = games_with_injuries[injury_cols].describe()
    print(summary)
    
    print("\nGames with injuries:")
    print(f"  Games with players out (home): {(games_with_injuries['players_out_home'] > 0).sum()}")
    print(f"  Games with players out (away): {(games_with_injuries['players_out_away'] > 0).sum()}")
    print(f"  Games with star out (home): {(games_with_injuries['star_out_home'] > 0).sum()}")
    print(f"  Games with star out (away): {(games_with_injuries['star_out_away'] > 0).sum()}")

print("\n" + "="*70)
print("✓ Injury feature analysis complete!")
print("="*70)
print("\nNext steps:")
print("1. Review correlation matrix above")
print("2. If correlations are high (>0.8), consider removing redundant features")
print("3. Re-run prediction pipeline with injury features")
print("4. Compare feature importance with/without injuries")

