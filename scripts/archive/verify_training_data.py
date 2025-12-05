#!/usr/bin/env python3
"""
Verify that the final dataset is ready for training
"""
import pandas as pd
import os
import sys

print("="*70)
print("TRAINING DATA VERIFICATION")
print("="*70)

# Check if file exists
data_file = 'final_dataset_with_injuries.csv'
if not os.path.exists(data_file):
    print(f"❌ File not found: {data_file}")
    sys.exit(1)

print(f"\n✓ Found: {data_file}")

# Load data
df = pd.read_csv(data_file, parse_dates=['date'])
print(f"✓ Loaded {len(df)} games")

# Check required columns
print("\n" + "="*70)
print("1. REQUIRED COLUMNS CHECK")
print("="*70)
required_cols = ['home_score', 'away_score', 'home_team_id', 'away_team_id', 'date']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"❌ Missing required columns: {missing}")
else:
    print("✓ All required columns present")

# Check data completeness
print("\n" + "="*70)
print("2. DATA COMPLETENESS")
print("="*70)
games_with_scores = (df['home_score'].notna() & df['away_score'].notna()).sum()
print(f"  Games with scores: {games_with_scores}/{len(df)} ({games_with_scores/len(df)*100:.1f}%)")
print(f"  Games with dates: {df['date'].notna().sum()}/{len(df)} ({df['date'].notna().sum()/len(df)*100:.1f}%)")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

# Check spread data
print("\n" + "="*70)
print("3. SPREAD DATA")
print("="*70)
opening_spreads = df['opening_spread'].notna().sum()
closing_spreads = df['closing_spread'].notna().sum()
print(f"  Opening spreads: {opening_spreads}/{len(df)} ({opening_spreads/len(df)*100:.1f}%)")
print(f"  Closing spreads: {closing_spreads}/{len(df)} ({closing_spreads/len(df)*100:.1f}%)")

if closing_spreads < len(df) * 0.5:
    print(f"  ⚠️  Note: Most games don't have historical spreads (expected)")
    print(f"     Model will use ELO-based estimates for games without spreads")

# Check injury features
print("\n" + "="*70)
print("4. INJURY FEATURES")
print("="*70)
injury_cols = [c for c in df.columns if any(x in c.lower() for x in ['injury', 'players_out', 'players_dtd', 'star_out'])]
print(f"  Found {len(injury_cols)} injury feature columns:")
for col in injury_cols:
    non_zero = (df[col] > 0).sum() if df[col].dtype in ['int64', 'float64'] else 0
    print(f"    - {col}: {non_zero} games with non-zero values")

# Check feature columns for model
print("\n" + "="*70)
print("5. MODEL FEATURES CHECK")
print("="*70)
expected_features = [
    'elo_home', 'elo_away', 'elo_diff',
    'rest_days_home', 'rest_days_away', 'rest_diff',
    'home_margin_avg_3', 'home_margin_avg_5', 'home_margin_avg_10',
    'away_margin_avg_3', 'away_margin_avg_5', 'away_margin_avg_10',
    'dk_spread', 'dk_spread_diff', 'dk_spread_move'
]

# These will be created during feature engineering, so we check if we have the base data
base_data_check = {
    'Has team IDs': 'home_team_id' in df.columns and 'away_team_id' in df.columns,
    'Has scores': 'home_score' in df.columns and 'away_score' in df.columns,
    'Has dates': 'date' in df.columns,
    'Has spreads (some)': 'opening_spread' in df.columns or 'closing_spread' in df.columns,
    'Has injury data': len(injury_cols) > 0
}

for check, result in base_data_check.items():
    status = "✓" if result else "❌"
    print(f"  {status} {check}")

# Final summary
print("\n" + "="*70)
print("✅ VERIFICATION SUMMARY")
print("="*70)

all_good = (
    len(missing) == 0 and
    games_with_scores == len(df) and
    len(injury_cols) > 0
)

if all_good:
    print("✓ Dataset is READY FOR TRAINING!")
    print(f"\n  Total games: {len(df)}")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Injury features: {len(injury_cols)}")
    print(f"  Spread data: {closing_spreads} games ({closing_spreads/len(df)*100:.1f}%)")
    print(f"\n  Next step: Run 'python3 scripts/run_full_pipeline.py' to train the model")
else:
    print("⚠️  Some issues detected - review above")

print("="*70)

