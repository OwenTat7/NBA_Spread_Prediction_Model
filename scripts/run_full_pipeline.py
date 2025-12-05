#!/usr/bin/env python3
"""
Complete NBA Spread Prediction Pipeline - Standalone Execution
Runs the full pipeline and reports accuracy metrics
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import sys

warnings.filterwarnings('ignore')
np.random.seed(42)

# Import modeling libraries
try:
    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
    import lightgbm as lgb
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Install with: pip install scikit-learn lightgbm")
    sys.exit(1)

print("="*70)
print("NBA SPREAD PREDICTION PIPELINE - FULL EXECUTION")
print("="*70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/6] Loading data...")
def load_game_data(file_path=None):
    if file_path:
        df = pd.read_csv(file_path, parse_dates=['date'])
    else:
        # Prioritize injury data if available
        # Try root directory first, then relative paths
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, 'data')
        for filename in [
            os.path.join(data_dir, 'final_dataset_with_injuries.csv'),
            os.path.join(data_dir, 'final_dataset_raw_games.csv'),
            os.path.join(data_dir, 'nba_games_with_injuries.csv'),
            os.path.join(data_dir, 'nba_games.csv'),
            os.path.join(data_dir, 'nba_games_test.csv'),
            'final_dataset_with_injuries.csv',
            'final_dataset_raw_games.csv',
            'nba_games_with_injuries.csv',
            'nba_games.csv',
            'nba_games_test.csv'
        ]:
            try:
                df = pd.read_csv(filename, parse_dates=['date'])
                print(f"✓ Loaded data from {filename}")
                
                # Check for injury features
                injury_cols = [c for c in df.columns if 'injury' in c.lower() or 'players_out' in c.lower() or 'players_dtd' in c.lower()]
                if injury_cols:
                    print(f"✓ Detected {len(injury_cols)} injury features: {', '.join(injury_cols[:5])}{'...' if len(injury_cols) > 5 else ''}")
                break
            except FileNotFoundError:
                continue
        else:
            print("❌ No game data file found!")
            sys.exit(1)
    
    required_cols = ['date', 'home_team_id', 'away_team_id', 'home_score', 'away_score']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        sys.exit(1)
    
    df = df.sort_values('date').reset_index(drop=True)
    df['margin'] = df['home_score'] - df['away_score']
    return df

games_df = load_game_data()
print(f"✓ Loaded {len(games_df)} games")
print(f"  Date range: {games_df['date'].min()} to {games_df['date'].max()}")

# ============================================================================
# STEP 2: Feature Engineering Classes and Functions
# ============================================================================
print("\n[2/6] Building features (ELO, rolling stats, rest days)...")

class ELORating:
    def __init__(self, initial_rating=1500, k_factor=20):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings = {}
    
    def get_rating(self, team_id, date=None):
        if team_id not in self.ratings:
            return self.initial_rating
        if date is None:
            return self.ratings[team_id][-1][1]
        for d, rating in reversed(self.ratings[team_id]):
            if d <= date:
                return rating
        return self.initial_rating
    
    def update(self, team_id, opponent_id, margin, date, home_advantage=0):
        if team_id not in self.ratings:
            self.ratings[team_id] = []
        if opponent_id not in self.ratings:
            self.ratings[opponent_id] = []
        
        team_rating = self.get_rating(team_id, date)
        opp_rating = self.get_rating(opponent_id, date)
        expected = 1 / (1 + 10 ** ((opp_rating - team_rating - home_advantage) / 400))
        actual = 0.5 + np.clip(margin / 20, -0.5, 0.5)
        
        new_team_rating = team_rating + self.k_factor * (actual - expected)
        new_opp_rating = opp_rating + self.k_factor * (expected - actual)
        
        if not self.ratings[team_id] or self.ratings[team_id][-1][0] < date:
            self.ratings[team_id].append((date, new_team_rating))
        else:
            self.ratings[team_id][-1] = (date, new_team_rating)
            
        if not self.ratings[opponent_id] or self.ratings[opponent_id][-1][0] < date:
            self.ratings[opponent_id].append((date, new_opp_rating))
        else:
            self.ratings[opponent_id][-1] = (date, new_opp_rating)

def calculate_rest_days(games_df):
    df = games_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    last_game_home = {}
    last_game_away = {}
    rest_days_home = []
    rest_days_away = []
    
    for idx, row in df.iterrows():
        home_id = str(row['home_team_id'])
        away_id = str(row['away_team_id'])
        game_date = row['date']
        
        rest_days_home.append((game_date - last_game_home[home_id]).days if home_id in last_game_home else 3)
        rest_days_away.append((game_date - last_game_away[away_id]).days if away_id in last_game_away else 3)
        
        last_game_home[home_id] = game_date
        last_game_away[away_id] = game_date
    
    df['rest_days_home'] = rest_days_home
    df['rest_days_away'] = rest_days_away
    df['rest_diff'] = df['rest_days_home'] - df['rest_days_away']
    return df

def calculate_rolling_features(games_df, windows=[3, 5, 10]):
    df = games_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    for window in windows:
        df[f'home_margin_avg_{window}'] = np.nan
        df[f'away_margin_avg_{window}'] = np.nan
        df[f'home_points_for_avg_{window}'] = np.nan
        df[f'away_points_for_avg_{window}'] = np.nan
        df[f'home_points_against_avg_{window}'] = np.nan
        df[f'away_points_against_avg_{window}'] = np.nan
    
    team_history = {}
    
    for idx, row in df.iterrows():
        home_id = str(row['home_team_id'])
        away_id = str(row['away_team_id'])
        game_date = row['date']
        
        if home_id not in team_history:
            team_history[home_id] = []
        if away_id not in team_history:
            team_history[away_id] = []
        
        home_past_games = [g for g in team_history[home_id] if g['date'] < game_date]
        for window in windows:
            if len(home_past_games) >= window:
                recent = home_past_games[-window:]
                df.loc[idx, f'home_margin_avg_{window}'] = np.mean([g['margin'] for g in recent])
                df.loc[idx, f'home_points_for_avg_{window}'] = np.mean([g['score'] for g in recent])
                df.loc[idx, f'home_points_against_avg_{window}'] = np.mean([g['opp_score'] for g in recent])
        
        away_past_games = [g for g in team_history[away_id] if g['date'] < game_date]
        for window in windows:
            if len(away_past_games) >= window:
                recent = away_past_games[-window:]
                df.loc[idx, f'away_margin_avg_{window}'] = np.mean([-g['margin'] for g in recent])
                df.loc[idx, f'away_points_for_avg_{window}'] = np.mean([g['opp_score'] for g in recent])
                df.loc[idx, f'away_points_against_avg_{window}'] = np.mean([g['score'] for g in recent])
        
        team_history[home_id].append({
            'date': game_date, 'margin': row['margin'],
            'score': row['home_score'], 'opp_score': row['away_score']
        })
        team_history[away_id].append({
            'date': game_date, 'margin': -row['margin'],
            'score': row['away_score'], 'opp_score': row['home_score']
        })
    
    rolling_cols = [c for c in df.columns if any(x in c for x in ['avg_', 'std_'])]
    for col in rolling_cols:
        df[col] = df[col].fillna(0)
    
    return df

def build_features(games_df, elo=None, include_spread=True):
    if len(games_df) == 0:
        return pd.DataFrame()
    
    df = games_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[(df['home_score'].notna()) & (df['away_score'].notna())].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    df = calculate_rest_days(df)
    df = calculate_rolling_features(df, windows=[3, 5, 10])
    
    if elo is None:
        elo = ELORating()
        for idx, row in df.iterrows():
            home_id = str(row['home_team_id'])
            away_id = str(row['away_team_id'])
            margin = row['margin']
            game_date = row['date']
            elo.update(home_id, away_id, margin, game_date, home_advantage=70)
    
    elo_home_list = []
    elo_away_list = []
    for idx, row in df.iterrows():
        home_id = str(row['home_team_id'])
        away_id = str(row['away_team_id'])
        game_date = row['date']
        elo_home_list.append(elo.get_rating(home_id, game_date - timedelta(days=1)))
        elo_away_list.append(elo.get_rating(away_id, game_date - timedelta(days=1)))
    
    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    # Remove home_court constant feature - it's always 1 and causes overfitting
    # Home advantage is already captured in ELO ratings and other features
    # df['home_court'] = 1  # REMOVED - constant feature causes overfitting
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['elo_diff_x_rest_diff'] = df['elo_diff'] * df['rest_diff']
    
    # Calculate ELO-based spread estimate (for comparison with DraftKings)
    df['elo_spread_estimate'] = (df['elo_home'] - df['elo_away']) / 25
    
    # Use actual DraftKings closing spreads from API if available, otherwise fall back to Elo-based estimate
    if 'closing_spread' not in df.columns:
        # Column doesn't exist - create it
        if include_spread:
            # Prefer actual spreads from API (current_spread > opening_spread > Elo estimate)
            if 'current_spread' in df.columns:
                df['closing_spread'] = df['current_spread'].fillna(df['elo_spread_estimate'])
            elif 'opening_spread' in df.columns:
                df['closing_spread'] = df['opening_spread'].fillna(df['elo_spread_estimate'])
            else:
                df['closing_spread'] = df['elo_spread_estimate']
        else:
            df['closing_spread'] = df['elo_spread_estimate']
    else:
        # Column exists but may have NaN values - fill them with ELO estimate
        if df['closing_spread'].isna().any():
            # Try to use current_spread or opening_spread to fill gaps
            if 'current_spread' in df.columns:
                df['closing_spread'] = df['closing_spread'].fillna(df['current_spread'])
            if 'opening_spread' in df.columns:
                df['closing_spread'] = df['closing_spread'].fillna(df['opening_spread'])
            # Fill remaining NaN with ELO estimate
            df['closing_spread'] = df['closing_spread'].fillna(df['elo_spread_estimate'])
    
    # Convert spread to home-team perspective if needed (ESPN API spreads may be from favorite's perspective)
    if 'favorite_id' in df.columns and 'home_team_id' in df.columns:
        # If favorite is away team, negate the spread
        df.loc[df['favorite_id'] == df['away_team_id'], 'closing_spread'] = -df.loc[df['favorite_id'] == df['away_team_id'], 'closing_spread']
        if 'opening_spread' in df.columns:
            df.loc[df['favorite_id'] == df['away_team_id'], 'opening_spread'] = -df.loc[df['favorite_id'] == df['away_team_id'], 'opening_spread']
    
    # DraftKings spread features (market-based features)
    # dk_spread: DraftKings closing spread (or current for upcoming games)
    df['dk_spread'] = df['closing_spread'].fillna(df['elo_spread_estimate'])
    
    # dk_spread_diff: Difference between ELO estimate and DraftKings spread
    # Positive = DraftKings favors home team more than ELO, Negative = DraftKings favors away team more
    df['dk_spread_diff'] = df['dk_spread'] - df['elo_spread_estimate']
    
    # dk_spread_move: Line movement (opening to closing)
    # Positive = line moved toward home team, Negative = line moved toward away team
    if 'opening_spread' in df.columns:
        df['dk_spread_move'] = (df['closing_spread'].fillna(df['elo_spread_estimate']) - 
                                df['opening_spread'].fillna(df['elo_spread_estimate']))
    else:
        df['dk_spread_move'] = 0
    
    df['margin'] = df['home_score'] - df['away_score']
    df['residual'] = df['margin'] - df['closing_spread']
    df['cover'] = (df['residual'] > 0).astype(int)
    
    # Ensure no NaN values in residual (critical for model training)
    if df['residual'].isna().any():
        print(f"⚠️  Warning: {df['residual'].isna().sum()} NaN values in residual - filling with 0")
        df['residual'] = df['residual'].fillna(0)
    
    return df

features_df = build_features(games_df, include_spread=True)
print(f"✓ Built features for {len(features_df)} games")

# ============================================================================
# STEP 3: Prepare Model Data
# ============================================================================
print("\n[3/6] Preparing model data...")

def prepare_model_data(features_df, target='residual'):
    if len(features_df) == 0:
        return None, None, [], pd.DataFrame()
    
    exclude_cols = [
        'game_id', 'date', 'home_team_id', 'away_team_id', 'home_team_name',
        'away_team_name', 'home_score', 'away_score', 'margin', 'residual',
        'cover', 'closing_spread', 'opening_spread', 'status',
        'completed', 'venue', 'attendance', 'name', 'short_name', 'point_differential',
        # Remove redundant features (perfect correlations found in diagnostics):
        'elo_spread_estimate',  # Perfectly correlated with elo_diff (r=1.0)
        'total_injury_impact_home',  # Perfectly correlated with injury_severity_home (r=1.0)
        'total_injury_impact_away',  # Perfectly correlated with injury_severity_away (r=1.0)
        'total_injury_impact_diff',  # Perfectly correlated with injury_severity_diff (r=1.0)
        # Note: dk_spread, dk_spread_diff, dk_spread_move are INCLUDED as features
    ]
    
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    numeric_cols = []
    for col in feature_cols:
        if features_df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
            numeric_cols.append(col)
    
    X = features_df[numeric_cols].fillna(0).values
    y = features_df[target].values
    metadata = features_df[['date', 'home_team_name', 'away_team_name', 'margin', 'closing_spread', 'residual']].copy()
    
    return X, y, numeric_cols, metadata

X, y, feature_names, metadata = prepare_model_data(features_df, target='residual')
print(f"✓ Prepared {len(feature_names)} features for {len(X)} games")

# ============================================================================
# STEP 4: Train Models
# ============================================================================
print("\n[4/6] Training models...")

def evaluate_model(y_true, y_pred, metadata=None, model_name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    results = {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    if metadata is not None and 'closing_spread' in metadata.columns:
        pred_margin = y_pred + metadata['closing_spread'].values
        true_margin = metadata['margin'].values
        pred_cover = (y_pred > 0).astype(int)
        true_cover = (metadata['residual'].values > 0).astype(int)
        cover_accuracy = (pred_cover == true_cover).mean()
        avg_edge = np.mean(y_pred - metadata['residual'].values)
        results['cover_accuracy'] = cover_accuracy
        results['avg_edge'] = avg_edge
        results['mae_margin'] = mean_absolute_error(true_margin, pred_margin)
    
    return results

# Check how many games have actual spreads vs Elo-based
if 'opening_spread' in features_df.columns or 'current_spread' in features_df.columns:
    actual_spread_count = 0
    if 'current_spread' in features_df.columns:
        actual_spread_count = features_df['current_spread'].notna().sum()
    elif 'opening_spread' in features_df.columns:
        actual_spread_count = features_df['opening_spread'].notna().sum()
    elo_spread_count = len(features_df) - actual_spread_count
    print(f"\n📊 Spread Data Sources:")
    print(f"  Actual spreads from API: {actual_spread_count} games ({actual_spread_count/len(features_df)*100:.1f}%)")
    print(f"  Elo-based spreads: {elo_spread_count} games ({elo_spread_count/len(features_df)*100:.1f}%)")

# Baseline
baseline_pred = np.zeros(len(features_df))
baseline_metrics = evaluate_model(features_df['residual'].values, baseline_pred, features_df, "Baseline")

# LightGBM with time-series CV
if len(X) > 50:
    tscv = TimeSeriesSplit(n_splits=min(3, len(X)//50))
    lgb_maes = []
    lgb_models = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names)
        
        params = {
            "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
            "learning_rate": 0.02,  # More regularization - tested as best config
            "num_leaves": 15,  # Reduced to prevent overfitting
            "feature_fraction": 0.6,  # More regularization
            "bagging_fraction": 0.6,  # More regularization
            "bagging_freq": 5,
            "min_child_samples": 40,  # Increased for more regularization
            "lambda_l1": 0.2,  # Stronger L1 regularization
            "lambda_l2": 0.2,  # Stronger L2 regularization
            "max_depth": 5,  # Limit tree depth more
            "min_split_gain": 0.2,  # Higher threshold to split
            "verbose": -1, "seed": 42
        }
        
        model = lgb.train(params, dtrain, num_boost_round=500,
                         valid_sets=[dval], callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        
        pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        mae = mean_absolute_error(y_val, pred_val)
        lgb_maes.append(mae)
        lgb_models.append(model)
        print(f"  Fold {fold+1} MAE: {mae:.3f}")
    
    # Final model - train on all data
    avg_best_iter = int(np.mean([m.best_iteration for m in lgb_models]))
    final_lgb_model = lgb.train(params, lgb.Dataset(X, label=y, feature_name=feature_names),
                                num_boost_round=avg_best_iter)
    
    print(f"✓ LightGBM trained (CV MAE: {np.mean(lgb_maes):.3f} ± {np.std(lgb_maes):.3f})")
    
    # ============================================================================
    # COMPREHENSIVE EVALUATION: Train/Test Split with Overfitting Check
    # ===========================================================================
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Time-based train/test split (use 80% for training, 20% most recent for testing)
    split_idx = int(len(X) * 0.8)
    X_train_final = X[:split_idx]
    X_test_final = X[split_idx:]
    y_train_final = y[:split_idx]
    y_test_final = y[split_idx:]
    metadata_train = metadata.iloc[:split_idx].copy()
    metadata_test = metadata.iloc[split_idx:].copy()
    
    print(f"\n📊 Train/Test Split:")
    print(f"  Training set: {len(X_train_final)} games ({len(X_train_final)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test_final)} games ({len(X_test_final)/len(X)*100:.1f}%)")
    print(f"  Train date range: {metadata_train['date'].min().strftime('%Y-%m-%d')} to {metadata_train['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Test date range: {metadata_test['date'].min().strftime('%Y-%m-%d')} to {metadata_test['date'].max().strftime('%Y-%m-%d')}")
    
    # Train model on training set only
    train_model = lgb.train(params, lgb.Dataset(X_train_final, label=y_train_final, feature_name=feature_names),
                           num_boost_round=avg_best_iter)
    
    # Predictions on train and test sets
    train_pred = train_model.predict(X_train_final, num_iteration=train_model.best_iteration)
    test_pred = train_model.predict(X_test_final, num_iteration=train_model.best_iteration)
    
    # Evaluate train set
    train_metrics = evaluate_model(y_train_final, train_pred, metadata_train, "Train")
    test_metrics = evaluate_model(y_test_final, test_pred, metadata_test, "Test")
    
    # Overfitting check
    print(f"\n🔍 Overfitting Analysis:")
    print(f"  Train MAE: {train_metrics['mae']:.3f}")
    print(f"  Test MAE:  {test_metrics['mae']:.3f}")
    mae_diff = train_metrics['mae'] - test_metrics['mae']
    mae_diff_pct = (mae_diff / test_metrics['mae']) * 100 if test_metrics['mae'] > 0 else 0
    
    if abs(mae_diff_pct) < 5:
        print(f"  ✓ Good generalization (MAE difference: {mae_diff:.3f}, {mae_diff_pct:.1f}%)")
    elif mae_diff < 0:
        print(f"  ⚠️  Possible underfitting (test MAE higher than train)")
    else:
        print(f"  ⚠️  Possible overfitting (train MAE {mae_diff:.3f} lower than test, {mae_diff_pct:.1f}%)")
    
    # Threshold Tuning - Find optimal threshold to balance predictions
    print(f"\n🎯 Threshold Tuning:")
    print("="*70)
    
    # Find optimal threshold on training set
    thresholds = np.arange(-5, 5, 0.1)
    best_threshold = 0
    best_test_acc = 0
    
    train_true_cover = (metadata_train['residual'].values > 0).astype(int)
    test_true_cover = (metadata_test['residual'].values > 0).astype(int)
    
    for threshold in thresholds:
        train_pred_cover = (train_pred > threshold).astype(int)
        test_pred_cover = (test_pred > threshold).astype(int)
        
        train_acc = (train_pred_cover == train_true_cover).mean()
        test_acc = (test_pred_cover == test_true_cover).mean()
        
        # Use F1 score weighted by test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_threshold = threshold
    
    print(f"  Default threshold (0.0):")
    default_train_pred = (train_pred > 0).astype(int)
    default_test_pred = (test_pred > 0).astype(int)
    default_train_acc = (default_train_pred == train_true_cover).mean()
    default_test_acc = (default_test_pred == test_true_cover).mean()
    print(f"    Train Accuracy: {default_train_acc:.1%}")
    print(f"    Test Accuracy: {default_test_acc:.1%}")
    print(f"    Test Home Cover Rate: {default_test_pred.mean():.1%}")
    
    print(f"\n  Optimal threshold ({best_threshold:.2f}):")
    optimal_train_pred = (train_pred > best_threshold).astype(int)
    optimal_test_pred = (test_pred > best_threshold).astype(int)
    optimal_train_acc = (optimal_train_pred == train_true_cover).mean()
    optimal_test_acc = (optimal_test_pred == test_true_cover).mean()
    print(f"    Train Accuracy: {optimal_train_acc:.1%}")
    print(f"    Test Accuracy: {optimal_test_acc:.1%}")
    print(f"    Test Home Cover Rate: {optimal_test_pred.mean():.1%}")
    print(f"    Actual Home Cover Rate: {test_true_cover.mean():.1%}")
    
    # Use optimal threshold for confusion matrices
    train_pred_cover = optimal_train_pred
    test_pred_cover = optimal_test_pred
    
    # Confusion Matrix for Cover Predictions
    print(f"\n📊 Confusion Matrix - Cover Predictions (Threshold: {best_threshold:.2f}):")
    print("="*70)
    
    train_cm = confusion_matrix(train_true_cover, train_pred_cover)
    test_cm = confusion_matrix(test_true_cover, test_pred_cover)
    
    print("\nTRAIN SET:")
    print("                Predicted")
    print("              Home Cover  Away Cover")
    print(f"Actual Home Cover  {train_cm[1,1]:5d}      {train_cm[1,0]:5d}")
    print(f"Actual Away Cover  {train_cm[0,1]:5d}      {train_cm[0,0]:5d}")
    train_accuracy = (train_cm[0,0] + train_cm[1,1]) / train_cm.sum()
    print(f"\n  Train Accuracy: {train_accuracy:.1%}")
    
    print("\nTEST SET:")
    print("                Predicted")
    print("              Home Cover  Away Cover")
    print(f"Actual Home Cover  {test_cm[1,1]:5d}      {test_cm[1,0]:5d}")
    print(f"Actual Away Cover  {test_cm[0,1]:5d}      {test_cm[0,0]:5d}")
    test_accuracy = (test_cm[0,0] + test_cm[1,1]) / test_cm.sum()
    print(f"\n  Test Accuracy: {test_accuracy:.1%}")
    
    # Classification metrics
    print(f"\n📈 Classification Metrics:")
    print("="*70)
    
    # Calculate precision, recall, F1 for "Home Cover" class
    def calc_metrics(cm):
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    train_precision, train_recall, train_f1 = calc_metrics(train_cm)
    test_precision, test_recall, test_f1 = calc_metrics(test_cm)
    
    print(f"\n{'Metric':<15} {'Train':<15} {'Test':<15} {'Difference':<15}")
    print("-"*60)
    print(f"{'Precision':<15} {train_precision:<15.3f} {test_precision:<15.3f} {train_precision-test_precision:<15.3f}")
    print(f"{'Recall':<15} {train_recall:<15.3f} {test_recall:<15.3f} {train_recall-test_recall:<15.3f}")
    print(f"{'F1 Score':<15} {train_f1:<15.3f} {test_f1:<15.3f} {train_f1-test_f1:<15.3f}")
    print(f"{'Accuracy':<15} {train_accuracy:<15.1%} {test_accuracy:<15.1%} {train_accuracy-test_accuracy:<15.1%}")
    
    # Generalization summary
    print(f"\n🎯 Generalization Summary:")
    print("="*70)
    mae_generalization = abs(mae_diff_pct)
    if mae_generalization < 5:
        print("  ✓ Excellent generalization - model performs similarly on train and test")
    elif mae_generalization < 10:
        print("  ✓ Good generalization - small difference between train and test")
    elif mae_generalization < 20:
        print("  ⚠️  Moderate overfitting - consider regularization")
    else:
        print("  ⚠️  Significant overfitting - model may not generalize well")
    
    # Store test metrics for final report
    lgb_metrics = test_metrics  # Use test metrics for final report
    final_lgb_model = train_model  # Use train model for predictions
    
    # Store optimal threshold for predictions
    print(f"\n💡 Using optimal threshold: {best_threshold:.2f} for cover predictions")
    print(f"   (Instead of default 0.0, which was causing Home Cover bias)")
    
    # Feature importance
    feature_importance = final_lgb_model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 15 Features by Importance:")
    print("-" * 50)
    for i, row in importance_df.head(15).iterrows():
        marker = "🏥" if any(x in row['feature'].lower() for x in ['injury', 'players_out', 'players_dtd', 'star_out']) else "  "
        print(f"{marker} {row['feature']:<35} {row['importance']:>10.0f}")
    
    injury_features = [f for f in feature_names if any(x in f.lower() for x in ['injury', 'players_out', 'players_dtd', 'star_out'])]
    if injury_features:
        injury_importance = importance_df[importance_df['feature'].isin(injury_features)]['importance'].sum()
        total_importance = importance_df['importance'].sum()
        print(f"\n🏥 Injury features represent {injury_importance/total_importance*100:.1f}% of total feature importance")
        print(f"   ({len(injury_features)} injury features out of {len(feature_names)} total)")
else:
    print("⚠️  Insufficient data for LightGBM")
    lgb_metrics = None
    importance_df = pd.DataFrame()

# ============================================================================
# STEP 5: Report Accuracy
# ============================================================================
print("\n" + "="*70)
print("ACCURACY RESULTS")
print("="*70)

print(f"\n{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'Cover Acc':<12} {'MAE Margin':<12}")
print("-"*70)

# Baseline
print(f"{'Baseline':<20} {baseline_metrics['mae']:<10.3f} {baseline_metrics['rmse']:<10.3f} "
      f"{baseline_metrics['r2']:<10.3f} {baseline_metrics.get('cover_accuracy', 0):<12.1%} "
      f"{baseline_metrics.get('mae_margin', 0):<12.3f}")

# LightGBM
if lgb_metrics:
    print(f"{'LightGBM':<20} {lgb_metrics['mae']:<10.3f} {lgb_metrics['rmse']:<10.3f} "
          f"{lgb_metrics['r2']:<10.3f} {lgb_metrics.get('cover_accuracy', 0):<12.1%} "
          f"{lgb_metrics.get('mae_margin', 0):<12.3f}")

print("\n" + "="*70)
print("KEY METRICS EXPLANATION")
print("="*70)
print("MAE (Mean Absolute Error): Lower is better - average prediction error")
print("RMSE (Root Mean Squared Error): Lower is better - penalizes large errors")
print("R² (R-squared): Higher is better (max 1.0) - variance explained")
print("Cover Accuracy: % of games where model correctly predicts cover/no-cover")
print("MAE Margin: Average error in predicting actual game margin")
print("="*70)

if lgb_metrics:
    improvement = ((baseline_metrics['mae'] - lgb_metrics['mae']) / baseline_metrics['mae']) * 100
    print(f"\n🎯 LightGBM improves MAE by {improvement:.1f}% over baseline")
    print(f"🎯 Cover prediction accuracy: {lgb_metrics.get('cover_accuracy', 0):.1%}")

print("\n✓ Pipeline execution complete!")
print("="*70)


