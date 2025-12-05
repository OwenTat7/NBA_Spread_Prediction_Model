# Data Collection Scripts - Complete Audit

## Overview
This document details what data each script collects, their dependencies, and identifies redundancies.

---

## 📊 DATA COLLECTION SCRIPTS

### 1. `fetch_historical_data.py` ⭐ **PRIMARY DATA COLLECTION**
**Purpose**: Fetch historical NBA games with scores and spreads

**Data Collected**:
- **From Scoreboard API** (`/scoreboard`):
  - `game_id` - ESPN game identifier
  - `date` - Game date
  - `name` - Game name/description
  - `status` - Game status
  - `completed` - Boolean completion flag
  - `venue` - Venue name
  - **Team Info**:
    - `home_team_id`, `home_team_name`, `home_team_abbrev`
    - `away_team_id`, `away_team_name`, `away_team_abbrev`
  - **Scores**:
    - `home_score`, `away_score`
    - `home_winner`, `away_winner`
    - `point_differential` (calculated)
  - **Spreads** (if available on scoreboard):
    - `opening_spread` - Opening line
    - `over_under` - Total points
    - `favorite_id` - Favorite team ID

- **From Summary API** (`/summary`) - For games missing spreads:
  - Fetches spread data for completed games that didn't have it on scoreboard
  - Converts to home-team perspective
  - Adds `opening_spread`, `over_under`, `favorite_id`

**Output**: `data/nba_games.csv`
**Time**: ~20-40 minutes (depends on date range)
**API Calls**: 
  - 1 per date (scoreboard)
  - 1 per game missing spread (summary)

---

### 2. `add_injuries_to_pipeline.py` ⭐ **INJURY DATA COLLECTION**
**Purpose**: Add injury features to existing game data

**Data Collected**:
- **From Summary API** (`/summary`):
  - Injury data for each game:
    - `players_out_home` / `players_out_away` - Count of players out
    - `players_dtd_home` / `players_dtd_away` - Count day-to-day
    - `injury_severity_home` / `injury_severity_away` - Severity scores
    - `star_out_home` / `star_out_away` - Binary (any star out)
    - `total_injury_impact_home` / `total_injury_impact_away` - Impact scores
    - `injury_severity_diff` - Home - Away severity
    - `players_out_diff` - Home - Away count
    - `total_injury_impact_diff` - Home - Away impact

**Dependencies**: Requires `nba_games.csv` (from script #1)
**Output**: `data/nba_games_with_injuries.csv`
**Time**: ~15-30 minutes (1 API call per game)
**API Calls**: 1 per game (summary endpoint)

**Note**: Preserves all existing columns (including spreads)

---

### 3. `injury_features.py` 📦 **MODULE**
**Purpose**: Helper functions for injury feature engineering

**Functions**:
- `extract_injuries_from_game()` - Parse injury data from API
- `calculate_injury_severity_score()` - Score injury impact
- `build_injury_features_for_game()` - Create feature dict
- `add_injury_features_to_games()` - Batch process games
- `analyze_injury_correlations()` - Correlation analysis
- `check_multicollinearity()` - Multicollinearity checks

**No direct data collection** - Used by script #2

---

### 4. `predict_upcoming_games.py` 🎯 **PREDICTION SCRIPT**
**Purpose**: Fetch upcoming games and make predictions

**Data Collected**:
- **From Scoreboard API** (`/scoreboard`):
  - Upcoming games (next 2 days)
  - Same fields as `fetch_historical_data.py` but for future games
  - **Spreads**:
    - `opening_spread` - Opening line
    - `current_spread` - Current/latest line (if available)
    - `over_under` - Total points
    - `favorite_id` - Favorite team ID

**Also Does**:
- Trains model on historical data
- Makes predictions
- Exports to Excel/CSV
- Tracks prediction history and correctness

**Output**: 
- `predictions_latest.xlsx` / `.csv`
- `predictions_history.csv` / `.xlsx`
- `prediction_correctness.csv` / `.xlsx`
- `predictions_summary.xlsx`

**Time**: ~2-5 minutes (fetches 2 days of games + model training)

---

### 5. `daily_predictions.py` 🔄 **WRAPPER**
**Purpose**: Simple wrapper to run `predict_upcoming_games.py`

**No data collection** - Just calls script #4

---

### 6. `run_full_pipeline.py` 🏋️ **MODEL TRAINING**
**Purpose**: Train and evaluate the prediction model

**Data Used** (doesn't collect):
- Reads `data/nba_games_with_injuries.csv`
- Builds features (ELO, rolling stats, rest days)
- Trains LightGBM model
- Reports accuracy metrics

**No API calls** - Pure model training

---

### 7. `run_full_data_collection.py` 🚀 **MASTER SCRIPT**
**Purpose**: Run complete data collection pipeline

**What it does**:
1. Runs `fetch_historical_data.py` (script #1)
2. Runs `add_injuries_to_pipeline.py` (script #2)
3. Verifies spreads were collected

**No direct data collection** - Orchestrates scripts #1 and #2

---

## 🔍 ANALYSIS/UTILITY SCRIPTS

### 8. `analyze_injury_features.py` 📈
**Purpose**: Analyze which injury features improve model performance

**No data collection** - Analyzes existing data

---

### 9. `fetch_historical_data_test.py` 🧪
**Purpose**: Quick test - fetch last 30 days

**Data Collected**: Same as `fetch_historical_data.py` but limited to 30 days
**Output**: `nba_games_test.csv`
**Note**: ⚠️ **REDUNDANT** - Can be removed, use `fetch_historical_data.py` instead

---

### 10. `fetch_historical_spreads.py` 🧪
**Purpose**: Test if ESPN API provides spreads for completed games

**Data Collected**: Tests spread extraction on 10 recent games
**Output**: Console output only
**Note**: ⚠️ **REDUNDANT** - Functionality now in `fetch_historical_data.py`

---

### 11. `backfill_historical_spreads.py` 🔧
**Purpose**: Backfill spreads for games that don't have them

**Data Collected**: Spreads from Summary API for games missing spreads
**Note**: ⚠️ **REDUNDANT** - `fetch_historical_data.py` now does this automatically

---

### 12. `check_spreads.py` ✅
**Purpose**: Check if spreads exist in data file

**No data collection** - Just reads and reports

---

### 13. `add_injuries_background.py` 🔄
**Purpose**: Non-interactive version of `add_injuries_to_pipeline.py`

**Data Collected**: Same as script #2
**Note**: ⚠️ **REDUNDANT** - Script #2 already supports non-interactive mode

---

### 14. `collect_injuries_background.py` 🔄
**Purpose**: Background injury collection

**Note**: ⚠️ **REDUNDANT** - Same as script #13

---

### 15. `test_prediction_pipeline.py` 🧪
**Purpose**: Quick test of prediction pipeline

**No data collection** - Tests with existing data

---

### 16. `run_prediction_pipeline.py` 🧪
**Purpose**: Simplified pipeline runner

**Note**: ⚠️ **REDUNDANT** - Use `run_full_pipeline.py` instead

---

## 📋 RECOMMENDED WORKFLOW

### For Initial Setup:
```bash
# Step 1: Fetch all historical games with spreads
python3 scripts/fetch_historical_data.py

# Step 2: Add injury features
python3 scripts/add_injuries_to_pipeline.py

# OR use master script:
python3 scripts/run_full_data_collection.py
```

### For Daily Predictions:
```bash
# Run daily (or via cron)
python3 scripts/daily_predictions.py
```

### For Model Training/Evaluation:
```bash
python3 scripts/run_full_pipeline.py
```

---

## 🗑️ SCRIPTS TO REMOVE (Redundant)

1. `fetch_historical_data_test.py` - Use `fetch_historical_data.py` with date range
2. `fetch_historical_spreads.py` - Functionality in `fetch_historical_data.py`
3. `backfill_historical_spreads.py` - Functionality in `fetch_historical_data.py`
4. `add_injuries_background.py` - Use `add_injuries_to_pipeline.py` with `--non-interactive`
5. `collect_injuries_background.py` - Same as above
6. `run_prediction_pipeline.py` - Use `run_full_pipeline.py`

---

## ✅ ESSENTIAL SCRIPTS (Keep)

1. `fetch_historical_data.py` - Primary data collection
2. `add_injuries_to_pipeline.py` - Injury features
3. `injury_features.py` - Injury module (used by #2)
4. `predict_upcoming_games.py` - Predictions
5. `daily_predictions.py` - Daily wrapper
6. `run_full_pipeline.py` - Model training
7. `run_full_data_collection.py` - Master collection script
8. `analyze_injury_features.py` - Analysis tool
9. `check_spreads.py` - Utility check

---

## 📊 DATA FLOW SUMMARY

```
ESPN API
  ├── Scoreboard Endpoint (/scoreboard)
  │   └── fetch_historical_data.py
  │       └── Basic game info + scores + spreads (if available)
  │
  └── Summary Endpoint (/summary)
      ├── fetch_historical_data.py (for missing spreads)
      ├── add_injuries_to_pipeline.py (for injuries)
      └── predict_upcoming_games.py (for completed game results)

Output Files:
  nba_games.csv (games + spreads)
    └── nba_games_with_injuries.csv (games + spreads + injuries)
        └── Used by run_full_pipeline.py (model training)
        └── Used by predict_upcoming_games.py (predictions)
```

---

## ⚡ EFFICIENCY NOTES

1. **Spread Collection**: Now automatic in `fetch_historical_data.py` - no need for separate backfill
2. **Injury Collection**: Can be slow (1 API call per game) - consider batching or caching
3. **Redundant Scripts**: Several test/background scripts can be consolidated
4. **API Rate Limiting**: All scripts use 0.3s delay between calls

---

## 🎯 RECOMMENDATIONS

1. **Remove redundant scripts** (listed above)
2. **Keep core pipeline**: fetch → add injuries → train → predict
3. **Consider caching**: Injury data doesn't change, could cache API responses
4. **Batch processing**: Already implemented in most scripts

