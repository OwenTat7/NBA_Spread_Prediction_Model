#!/usr/bin/env python3
"""
Script to fetch historical NBA game data from ESPN API
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sys

NBA_BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba"

def fetch_api_data(url, params=None, max_retries=3):
    """Fetch data from ESPN API with error handling and retry logic"""
    for attempt in range(max_retries):
        try:
            # Increase timeout and add retry delay
            timeout = 30 if attempt > 0 else 15
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                time.sleep(wait_time)
                continue
            print(f"Error fetching {url}: Timeout after {max_retries} attempts")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"Error fetching {url}: {e}")
            return None
    return None

def fetch_game_summary(game_id, max_retries=3):
    """Fetch game summary to get spread data for completed games"""
    url = f"{NBA_BASE_URL}/summary"
    params = {'event': str(game_id)}
    for attempt in range(max_retries):
        try:
            timeout = 30 if attempt > 0 else 15
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
                continue
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    return None

def extract_spread_from_summary(summary_data, home_team_id, away_team_id):
    """Extract spread/odds data from game summary and convert to home-team perspective
    
    Returns opening and closing spreads from DraftKings (or other provider)
    """
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
    
    # Extract favorite_id
    favorite_id = None
    if odds_data.get('favorite'):
        favorite_id = odds_data.get('favorite', {}).get('teamId') if isinstance(odds_data.get('favorite'), dict) else odds_data.get('favorite')
    elif odds_data.get('homeTeamOdds', {}).get('favorite'):
        favorite_id = odds_data.get('homeTeamOdds', {}).get('teamId')
    elif odds_data.get('awayTeamOdds', {}).get('favorite'):
        favorite_id = odds_data.get('awayTeamOdds', {}).get('teamId')
    
    # Extract opening and closing spreads from pointSpread object
    opening_spread = None
    closing_spread = None
    
    point_spread = odds_data.get('pointSpread', {})
    if point_spread:
        # Opening spread
        home_open = point_spread.get('home', {}).get('open', {})
        if home_open and home_open.get('line'):
            try:
                opening_spread = float(home_open.get('line').replace('-', '').replace('+', ''))
                if home_open.get('line').startswith('-'):
                    opening_spread = -opening_spread
            except:
                pass
        
        # Closing spread (preferred for historical games)
        home_close = point_spread.get('home', {}).get('close', {})
        if home_close and home_close.get('line'):
            try:
                closing_spread = float(home_close.get('line').replace('-', '').replace('+', ''))
                if home_close.get('line').startswith('-'):
                    closing_spread = -closing_spread
            except:
                pass
    
    # Fallback to simple spread field if pointSpread structure not available
    if closing_spread is None and opening_spread is None:
        spread_value = odds_data.get('spread')
        if spread_value is not None:
            closing_spread = spread_value
            opening_spread = spread_value  # Use same value if we can't distinguish
    
    # Convert to home-team perspective
    # If favorite is away team, negate the spreads
    if favorite_id and str(favorite_id) == str(away_team_id):
        if opening_spread is not None:
            opening_spread = -opening_spread
        if closing_spread is not None:
            closing_spread = -closing_spread
    
    result = {
        'over_under': odds_data.get('overUnder'),
        'favorite_id': favorite_id,
    }
    
    if opening_spread is not None:
        result['opening_spread'] = opening_spread
    if closing_spread is not None:
        result['closing_spread'] = closing_spread
    
    return result if (opening_spread is not None or closing_spread is not None) else None

def fetch_historical_games(start_date, end_date, output_file='final_dataset_raw_games.csv'):
    """Fetch games for a date range"""
    print(f"Fetching NBA games from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate date list
    date_list = []
    current = start_date
    while current <= end_date:
        date_list.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    print(f"Total days to process: {len(date_list)}")
    
    historical_games = []
    batch_size = 30
    total_batches = (len(date_list) + batch_size - 1) // batch_size
    
    # Try to use tqdm for progress bar, fallback to simple counter
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    if use_tqdm:
        batch_iterator = tqdm(range(0, len(date_list), batch_size), desc="Fetching games", unit="batch")
    else:
        batch_iterator = range(0, len(date_list), batch_size)
    
    for batch_num in batch_iterator:
        batch_dates = date_list[batch_num:batch_num + batch_size]
        batch_num_display = (batch_num // batch_size) + 1
        
        if not use_tqdm:
            print(f"Batch {batch_num_display}/{total_batches} ({len(batch_dates)} dates)...", end=" ")
        
        games_in_batch = 0
        for date in batch_dates:
            url = f"{NBA_BASE_URL}/scoreboard"
            params = {'dates': date}
            data = fetch_api_data(url, params)
            
            if data and 'events' in data:
                for event in data['events']:
                    game_info = {
                        'game_id': event.get('id'),
                        'date': date,
                        'name': event.get('name'),
                        'status': event.get('status', {}).get('type', {}).get('description'),
                        'completed': event.get('status', {}).get('type', {}).get('completed', False),
                    }
                    
                    competition = event.get('competitions', [{}])[0]
                    game_info['venue'] = competition.get('venue', {}).get('fullName')
                    
                    competitors = competition.get('competitors', [])
                    if len(competitors) >= 2:
                        home_team = competitors[0] if competitors[0].get('homeAway') == 'home' else competitors[1]
                        away_team = competitors[1] if competitors[0].get('homeAway') == 'home' else competitors[0]
                        
                        game_info['home_team_id'] = home_team.get('team', {}).get('id')
                        game_info['home_team_name'] = home_team.get('team', {}).get('displayName')
                        game_info['home_team_abbrev'] = home_team.get('team', {}).get('abbreviation')
                        game_info['home_score'] = home_team.get('score')
                        game_info['home_winner'] = home_team.get('winner', False)
                        
                        game_info['away_team_id'] = away_team.get('team', {}).get('id')
                        game_info['away_team_name'] = away_team.get('team', {}).get('displayName')
                        game_info['away_team_abbrev'] = away_team.get('team', {}).get('abbreviation')
                        game_info['away_score'] = away_team.get('score')
                        game_info['away_winner'] = away_team.get('winner', False)
                        
                        if game_info['home_score'] and game_info['away_score']:
                            try:
                                home_score = float(game_info['home_score'])
                                away_score = float(game_info['away_score'])
                                game_info['point_differential'] = home_score - away_score
                            except (ValueError, TypeError):
                                game_info['point_differential'] = None
                        
                        # Try to get spread from scoreboard odds (if available)
                        odds = competition.get('odds', [])
                        if odds:
                            spread_info = odds[0]
                            # Extract opening and closing spreads from pointSpread if available
                            point_spread = spread_info.get('pointSpread', {})
                            if point_spread:
                                # Opening spread
                                home_open = point_spread.get('home', {}).get('open', {})
                                if home_open and home_open.get('line'):
                                    try:
                                        opening_val = float(home_open.get('line').replace('-', '').replace('+', ''))
                                        if home_open.get('line').startswith('-'):
                                            opening_val = -opening_val
                                        game_info['opening_spread'] = opening_val
                                    except:
                                        game_info['opening_spread'] = spread_info.get('spread')
                                
                                # Closing spread
                                home_close = point_spread.get('home', {}).get('close', {})
                                if home_close and home_close.get('line'):
                                    try:
                                        closing_val = float(home_close.get('line').replace('-', '').replace('+', ''))
                                        if home_close.get('line').startswith('-'):
                                            closing_val = -closing_val
                                        game_info['closing_spread'] = closing_val
                                    except:
                                        game_info['closing_spread'] = spread_info.get('spread')
                                else:
                                    # Use current spread as closing if no close line
                                    game_info['closing_spread'] = spread_info.get('spread')
                            else:
                                # Fallback to simple spread field
                                game_info['opening_spread'] = spread_info.get('spread')
                                game_info['closing_spread'] = spread_info.get('spread')
                            
                            game_info['over_under'] = spread_info.get('overUnder')
                            favorite = spread_info.get('favorite', {})
                            if isinstance(favorite, dict):
                                game_info['favorite_id'] = favorite.get('teamId')
                            else:
                                game_info['favorite_id'] = favorite
                            
                            # Convert to home-team perspective if favorite is away team
                            if game_info.get('favorite_id') and str(game_info['favorite_id']) == str(game_info.get('away_team_id')):
                                if game_info.get('opening_spread') is not None:
                                    game_info['opening_spread'] = -game_info['opening_spread']
                                if game_info.get('closing_spread') is not None:
                                    game_info['closing_spread'] = -game_info['closing_spread']
                        
                        historical_games.append(game_info)
                        if game_info.get('home_score'):
                            games_in_batch += 1
            
            time.sleep(0.3)  # Rate limiting
        
        if not use_tqdm:
            print(f"✓ {games_in_batch} completed games")
        
        # Save progress every 5 batches
        if batch_num_display % 5 == 0 and historical_games:
            temp_df = pd.DataFrame(historical_games)
            temp_df.to_csv('nba_games_temp.csv', index=False)
            print(f"  (Progress saved: {len(historical_games)} total games)")
    
    # Create final DataFrame
    df = pd.DataFrame(historical_games)
    
    # Filter to completed games
    completed = df[
        (df['completed'] == True) & 
        (df['home_score'].notna()) & 
        (df['away_score'].notna())
    ].copy()
    
    # Convert date
    completed['date'] = pd.to_datetime(completed['date'], format='%Y%m%d')
    completed = completed.sort_values('date').reset_index(drop=True)
    
    # Fetch spreads for completed games that don't have them
    print(f"\n📊 Fetching spreads for completed games...")
    if 'opening_spread' not in completed.columns:
        completed['opening_spread'] = None
    if 'closing_spread' not in completed.columns:
        completed['closing_spread'] = None
    if 'over_under' not in completed.columns:
        completed['over_under'] = None
    if 'favorite_id' not in completed.columns:
        completed['favorite_id'] = None
    
    games_without_spreads = completed[completed['opening_spread'].isna() | completed['closing_spread'].isna()]
    print(f"   {len(games_without_spreads)} games need spread data")
    
    # Add progress bar for spread fetching
    if use_tqdm and len(games_without_spreads) > 0:
        spread_iterator = tqdm(games_without_spreads.iterrows(), desc="Fetching spreads", total=len(games_without_spreads), unit="game")
    else:
        spread_iterator = games_without_spreads.iterrows()
    
    for idx, row in spread_iterator:
        game_id = row['game_id']
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        summary = fetch_game_summary(game_id)
        spread_info = extract_spread_from_summary(summary, home_id, away_id)
        
        if spread_info:
            if 'opening_spread' in spread_info:
                completed.loc[idx, 'opening_spread'] = spread_info['opening_spread']
            if 'closing_spread' in spread_info:
                completed.loc[idx, 'closing_spread'] = spread_info['closing_spread']
            completed.loc[idx, 'over_under'] = spread_info.get('over_under')
            completed.loc[idx, 'favorite_id'] = spread_info.get('favorite_id')
        
        if not use_tqdm and (idx + 1) % 50 == 0:
            print(f"   Progress: {idx + 1}/{len(games_without_spreads)} games processed")
        
        time.sleep(0.3)  # Rate limiting
    
    spread_count = completed['opening_spread'].notna().sum()
    print(f"   ✓ Spreads fetched for {spread_count}/{len(completed)} games ({spread_count/len(completed)*100:.1f}%)")
    
    # Save
    completed.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Data Collection Complete!")
    print(f"{'='*60}")
    print(f"Total games fetched: {len(df)}")
    print(f"Completed games: {len(completed)}")
    if len(completed) > 0:
        print(f"Date range: {completed['date'].min().strftime('%Y-%m-%d')} to {completed['date'].max().strftime('%Y-%m-%d')}")
    print(f"Saved to: {output_file}")
    
    return completed

if __name__ == "__main__":
    # Fetch 2023-24 season (Oct 2023 to June 2024)
    # And current season (Oct 2024 onwards)
    start_date = datetime(2023, 10, 1)
    today = datetime.now()
    
    # End date: June 2024 for 2023-24 season, or today if we're in 2024-25 season
    if today > datetime(2024, 10, 1):
        # We're in 2024-25 season, fetch up to today
        end_date = today
    else:
        # Only fetch 2023-24 season
        end_date = min(datetime(2024, 6, 30), today)
    
    print("Starting NBA historical data collection...")
    print("This will take several minutes. Please be patient.\n")
    
    try:
        games_df = fetch_historical_games(start_date, end_date)
        print(f"\n✓ Successfully collected {len(games_df)} games")
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during collection: {e}")
        sys.exit(1)

