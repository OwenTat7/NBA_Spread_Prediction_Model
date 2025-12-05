#!/usr/bin/env python3
"""
Injury Feature Engineering Module
Extracts injury data from ESPN API and creates features for the prediction model
Includes multicollinearity checks and correlation analysis
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

NBA_BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba"

def fetch_api_data(url: str, params: Optional[Dict] = None) -> Dict:
    """Fetch data from ESPN API with error handling"""
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def extract_injuries_from_game(game_id: str) -> Dict:
    """
    Extract injury data from a game summary
    Returns: {team_id: [list of injured players]}
    """
    url = f"{NBA_BASE_URL}/summary"
    params = {'event': game_id}
    data = fetch_api_data(url, params)
    
    if not data or 'injuries' not in data:
        return {}
    
    injuries_by_team = {}
    
    for team_injuries in data['injuries']:
        team_id = team_injuries.get('team', {}).get('id')
        if not team_id:
            continue
        
        injuries = team_injuries.get('injuries', [])
        injury_list = []
        
        for injury in injuries:
            athlete = injury.get('athlete', {})
            status = injury.get('status', '')
            injury_type = injury.get('type', {})
            details = injury.get('details', {})
            
            injury_info = {
                'player_id': athlete.get('id'),
                'player_name': athlete.get('displayName', ''),
                'position': athlete.get('position', {}).get('abbreviation', ''),
                'status': status,  # 'Out', 'Day-To-Day', etc.
                'injury_type': injury_type.get('abbreviation', ''),
                'injury_description': details.get('type', ''),
                'return_date': details.get('returnDate', ''),
                'fantasy_status': details.get('fantasyStatus', {}).get('abbreviation', '')
            }
            injury_list.append(injury_info)
        
        injuries_by_team[team_id] = injury_list
    
    return injuries_by_team

def calculate_injury_severity_score(injury: Dict) -> float:
    """
    Calculate a severity score for an injury
    Higher score = more impactful
    """
    status = injury.get('status', '').upper()
    fantasy_status = injury.get('fantasy_status', '').upper()
    
    # Base score from status
    if 'OUT' in status or fantasy_status == 'OUT':
        base_score = 1.0
    elif 'DAY-TO-DAY' in status or 'GTD' in fantasy_status:
        base_score = 0.5
    else:
        base_score = 0.2
    
    # Position multiplier (stars/starters matter more)
    position = injury.get('position', '')
    if position in ['PG', 'SG', 'SF', 'PF', 'C']:  # All positions are important
        position_mult = 1.0
    else:
        position_mult = 0.8
    
    # Injury type multiplier
    injury_type = injury.get('injury_description', '').upper()
    if any(x in injury_type for x in ['SURGERY', 'FRACTURE', 'TORN', 'RUPTURE']):
        severity_mult = 1.5
    elif any(x in injury_type for x in ['SPRAIN', 'STRAIN']):
        severity_mult = 1.2
    else:
        severity_mult = 1.0
    
    return base_score * position_mult * severity_mult

def build_injury_features_for_game(game_id: str, home_team_id: str, away_team_id: str) -> Dict:
    """
    Build injury features for a single game
    Returns dictionary of features
    """
    injuries_by_team = extract_injuries_from_game(game_id)
    
    features = {
        'players_out_home': 0,
        'players_out_away': 0,
        'players_dtd_home': 0,  # Day-to-day
        'players_dtd_away': 0,
        'injury_severity_home': 0.0,
        'injury_severity_away': 0.0,
        'star_out_home': 0,  # Binary: any significant player out
        'star_out_away': 0,
        'total_injury_impact_home': 0.0,
        'total_injury_impact_away': 0.0
    }
    
    # Process home team injuries
    home_injuries = injuries_by_team.get(str(home_team_id), [])
    for injury in home_injuries:
        status = injury.get('status', '').upper()
        severity = calculate_injury_severity_score(injury)
        
        if 'OUT' in status:
            features['players_out_home'] += 1
            features['star_out_home'] = 1 if severity > 0.8 else features['star_out_home']
        elif 'DAY-TO-DAY' in status:
            features['players_dtd_home'] += 1
        
        features['injury_severity_home'] += severity
        features['total_injury_impact_home'] += severity
    
    # Process away team injuries
    away_injuries = injuries_by_team.get(str(away_team_id), [])
    for injury in away_injuries:
        status = injury.get('status', '').upper()
        severity = calculate_injury_severity_score(injury)
        
        if 'OUT' in status:
            features['players_out_away'] += 1
            features['star_out_away'] = 1 if severity > 0.8 else features['star_out_away']
        elif 'DAY-TO-DAY' in status:
            features['players_dtd_away'] += 1
        
        features['injury_severity_away'] += severity
        features['total_injury_impact_away'] += severity
    
    # Calculate differentials
    features['injury_severity_diff'] = features['injury_severity_home'] - features['injury_severity_away']
    features['players_out_diff'] = features['players_out_home'] - features['players_out_away']
    features['total_injury_impact_diff'] = features['total_injury_impact_home'] - features['total_injury_impact_away']
    
    return features

def add_injury_features_to_games(games_df: pd.DataFrame, batch_size: int = 10, delay: float = 0.5) -> pd.DataFrame:
    """
    Add injury features to a games DataFrame
    Fetches injury data from ESPN API for each game
    """
    df = games_df.copy()
    
    # Initialize injury feature columns
    injury_cols = [
        'players_out_home', 'players_out_away',
        'players_dtd_home', 'players_dtd_away',
        'injury_severity_home', 'injury_severity_away',
        'star_out_home', 'star_out_away',
        'total_injury_impact_home', 'total_injury_impact_away',
        'injury_severity_diff', 'players_out_diff', 'total_injury_impact_diff'
    ]
    
    for col in injury_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    print(f"Fetching injury data for {len(df)} games...")
    print("This may take several minutes due to API rate limiting...")
    
    # Try to use tqdm for progress bar, fallback to simple counter
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    if use_tqdm:
        batch_iterator = tqdm(range(0, len(df), batch_size), desc="Adding injuries", unit="batch", total=total_batches)
    else:
        batch_iterator = range(0, len(df), batch_size)
    
    for batch_num in batch_iterator:
        batch = df.iloc[batch_num:batch_num + batch_size]
        batch_num_display = (batch_num // batch_size) + 1
        
        if not use_tqdm:
            print(f"Processing batch {batch_num_display}/{total_batches}...", end=" ")
        
        for idx, row in batch.iterrows():
            game_id = row.get('game_id')
            home_team_id = str(row.get('home_team_id', ''))
            away_team_id = str(row.get('away_team_id', ''))
            
            if not game_id or not home_team_id or not away_team_id:
                continue
            
            try:
                injury_features = build_injury_features_for_game(
                    str(game_id), home_team_id, away_team_id
                )
                
                # Update DataFrame
                for col, value in injury_features.items():
                    df.loc[idx, col] = value
                
            except Exception as e:
                # If error, leave features as 0
                pass
            
            time.sleep(delay)  # Rate limiting
        
        if not use_tqdm:
            print(f"✓")
    
    print(f"\n✓ Injury features added to {len(df)} games")
    return df

def analyze_injury_correlations(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlations between injury features and spread-related features
    Returns correlation matrix
    """
    injury_cols = [c for c in features_df.columns if any(x in c for x in [
        'injury', 'players_out', 'star_out', 'players_dtd'
    ])]
    
    spread_cols = [c for c in features_df.columns if any(x in c for x in [
        'spread', 'line', 'odds', 'elo_diff'
    ])]
    
    if not injury_cols or not spread_cols:
        print("⚠️  Not enough columns for correlation analysis")
        return pd.DataFrame()
    
    # Calculate correlations
    all_cols = injury_cols + spread_cols
    numeric_df = features_df[all_cols].select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return pd.DataFrame()
    
    corr_matrix = numeric_df.corr()
    
    # Extract injury-spread correlations
    injury_spread_corr = corr_matrix.loc[injury_cols, spread_cols]
    
    return injury_spread_corr

def check_multicollinearity(features_df: pd.DataFrame, threshold: float = 0.8) -> Dict:
    """
    Check for multicollinearity between injury and spread features
    Returns dictionary with findings
    """
    corr_matrix = analyze_injury_correlations(features_df)
    
    if corr_matrix.empty:
        return {'high_correlations': [], 'warnings': []}
    
    high_correlations = []
    warnings = []
    
    for injury_col in corr_matrix.index:
        for spread_col in corr_matrix.columns:
            corr_value = abs(corr_matrix.loc[injury_col, spread_col])
            if corr_value > threshold:
                high_correlations.append({
                    'injury_feature': injury_col,
                    'spread_feature': spread_col,
                    'correlation': corr_value
                })
                warnings.append(
                    f"⚠️  High correlation ({corr_value:.2f}) between {injury_col} and {spread_col}"
                )
    
    return {
        'high_correlations': high_correlations,
        'warnings': warnings,
        'correlation_matrix': corr_matrix
    }

print("✓ Injury feature engineering module loaded")


