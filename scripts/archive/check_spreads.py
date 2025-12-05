#!/usr/bin/env python3
"""
Check if historical spreads have been backfilled
"""
import pandas as pd
import os
import sys

# Try to find the data file
possible_paths = [
    'data/nba_games_with_injuries.csv',
    'data/nba_games.csv',
    '../data/nba_games_with_injuries.csv',
    '../data/nba_games.csv',
]

data_file = None
for path in possible_paths:
    if os.path.exists(path):
        data_file = path
        break

if not data_file:
    print("❌ Could not find data file. Please check the file location.")
    print("   Looking for: data/nba_games_with_injuries.csv or data/nba_games.csv")
    sys.exit(1)

print(f"✓ Found data file: {data_file}")
df = pd.read_csv(data_file, parse_dates=['date'])

print(f"\n📊 Spread Data Status:")
print(f"   Total games: {len(df)}")
print(f"   Completed games: {df['completed'].sum() if 'completed' in df.columns else 'N/A'}")

if 'opening_spread' in df.columns:
    spread_count = df['opening_spread'].notna().sum()
    pct = (spread_count / len(df)) * 100 if len(df) > 0 else 0
    print(f"\n✅ Spread column exists!")
    print(f"   Games with spreads: {spread_count} / {len(df)} ({pct:.1f}%)")
    
    if spread_count > 0:
        print(f"\n   Sample spreads:")
        sample = df[df['opening_spread'].notna()].head(5)
        for idx, row in sample.iterrows():
            print(f"     {row['date']} - {row.get('away_team_abbrev', 'AWAY')} @ {row.get('home_team_abbrev', 'HOME')}: {row['opening_spread']}")
        
        if spread_count == len(df):
            print(f"\n🎉 All games have spreads backfilled!")
        else:
            print(f"\n⚠️  Only {pct:.1f}% of games have spreads. Run backfill script to add more.")
    else:
        print(f"\n⚠️  No spreads found. Run backfill script to add them.")
else:
    print(f"\n❌ No 'opening_spread' column found.")
    print(f"   Run the backfill script to add spreads to historical data.")

