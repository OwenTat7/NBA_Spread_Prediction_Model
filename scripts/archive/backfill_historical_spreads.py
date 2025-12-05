#!/usr/bin/env python3
"""
Backfill historical spread data from ESPN API for all completed games
"""
import pandas as pd
import requests
import time
from datetime import datetime
import sys
import os

NBA_BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba"

def fetch_game_summary(game_id):
    """Fetch game summary from ESPN API"""
    url = f"{NBA_BASE_URL}/summary"
    params = {'event': str(game_id)}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def extract_spread_from_summary(summary_data):
    """Extract spread/odds data from game summary"""
    if not summary_data:
        return None
    
    odds_data = None
    
    # Try header -> competitions -> odds
    header = summary_data.get('header', {})
    competitions = header.get('competitions', [])
    if competitions:
        odds = competitions[0].get('odds', [])
        if odds:
            odds_data = odds[0]
    
    # Try pickcenter (might be a list or dict)
    if not odds_data:
        pickcenter = summary_data.get('pickcenter', {})
        if pickcenter:
            if isinstance(pickcenter, list) and len(pickcenter) > 0:
                odds_data = pickcenter[0]
            elif isinstance(pickcenter, dict):
                odds_list = pickcenter.get('odds', [])
                if odds_list:
                    odds_data = odds_list[0]
    
    if not odds_data:
        return None
    
    # Extract spread information
    spread = odds_data.get('spread')
    favorite_id = None
    if odds_data.get('favorite'):
        favorite_id = odds_data.get('favorite', {}).get('teamId') if isinstance(odds_data.get('favorite'), dict) else odds_data.get('favorite')
    
    return {
        'spread': spread,
        'favorite_id': favorite_id,
        'over_under': odds_data.get('overUnder'),
    }

# Load historical data (same logic as run_full_pipeline.py)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root_dir, 'data')

data_file = None
for filename in [
    os.path.join(data_dir, 'nba_games_with_injuries.csv'),
    os.path.join(data_dir, 'nba_games.csv'),
    os.path.join(data_dir, 'nba_games_test.csv'),
    'nba_games_with_injuries.csv',
    'nba_games.csv',
    'nba_games_test.csv'
]:
    try:
        if os.path.exists(filename):
            data_file = filename
            break
    except:
        continue

if not data_file:
    print(f"❌ Data file not found!")
    print(f"   Tried: {data_dir}/nba_games_with_injuries.csv, {data_dir}/nba_games.csv, etc.")
    sys.exit(1)

df = pd.read_csv(data_file, parse_dates=['date'])
print(f"✓ Loaded {len(df)} games from {data_file}")

# Check if spreads already exist
if 'opening_spread' in df.columns and df['opening_spread'].notna().sum() > 0:
    print(f"⚠️  Found existing spread data for {df['opening_spread'].notna().sum()} games")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

# Filter to completed games
completed_games = df[df['completed'] == True].copy()
print(f"\n📊 Backfilling spreads for {len(completed_games)} completed games...")
print("   This will take approximately 30-60 minutes due to rate limiting.\n")

spreads_added = 0
spreads_skipped = 0
errors = 0

for idx, row in completed_games.iterrows():
    game_id = row['game_id']
    game_date = row['date']
    matchup = f"{row['away_team_abbrev']} @ {row['home_team_abbrev']}"
    
    # Skip if already has spread
    if 'opening_spread' in df.columns and pd.notna(df.loc[idx, 'opening_spread']):
        spreads_skipped += 1
        if (spreads_skipped + spreads_added) % 100 == 0:
            print(f"  Progress: {spreads_added} added, {spreads_skipped} skipped, {errors} errors")
        continue
    
    summary = fetch_game_summary(game_id)
    spread_info = extract_spread_from_summary(summary)
    
    if spread_info and spread_info.get('spread') is not None:
        spread_value = spread_info['spread']
        favorite_id = spread_info.get('favorite_id')
        
        # Convert to home-team perspective
        # If favorite is away team, negate the spread
        if favorite_id and favorite_id == row['away_team_id']:
            home_spread = -spread_value
        else:
            home_spread = spread_value
        
        # Add to dataframe
        if 'opening_spread' not in df.columns:
            df['opening_spread'] = None
            df['over_under'] = None
            df['favorite_id'] = None
        
        df.loc[idx, 'opening_spread'] = home_spread
        df.loc[idx, 'over_under'] = spread_info.get('over_under')
        df.loc[idx, 'favorite_id'] = favorite_id
        
        spreads_added += 1
        
        if spreads_added % 50 == 0:
            print(f"  Progress: {spreads_added} spreads added, {spreads_skipped} skipped, {errors} errors")
            # Save progress every 50 games
            df.to_csv(data_file, index=False)
    else:
        errors += 1
    
    time.sleep(0.3)  # Rate limiting

# Final save
df.to_csv(data_file, index=False)

print("\n" + "="*70)
print(f"✅ Backfill complete!")
print(f"   Spreads added: {spreads_added}")
print(f"   Already had spreads: {spreads_skipped}")
print(f"   Errors/not found: {errors}")
print(f"   Total games with spreads: {df['opening_spread'].notna().sum() if 'opening_spread' in df.columns else 0}")
print(f"\n   Data saved to: {data_file}")
print("="*70)



