# Data Collection Scripts - Organization Guide

## Overview
All data collection scripts have been reorganized with clear prefixes for easy identification.

## Scripts

### Main Data Collection Pipeline

#### `00_data_collection_run_all.py`
**Purpose**: Master script that runs the complete data collection pipeline  
**What it does**:
1. Fetches historical games with DraftKings spreads
2. Adds injury features to games
3. Produces final dataset ready for training

**Usage**:
```bash
python3 scripts/00_data_collection_run_all.py
```

**Output Files**:
- `data/final_dataset_raw_games.csv` - Games with spreads (intermediate)
- `data/final_dataset_with_injuries.csv` - **FINAL DATASET** (ready for training)

---

### Individual Data Collection Steps

#### `01_data_collection_fetch_historical_games.py`
**Purpose**: Fetches historical NBA games from ESPN API with DraftKings spreads  
**What it does**:
- Collects game data (scores, teams, dates)
- Extracts opening and closing spreads from DraftKings
- Saves to `data/final_dataset_raw_games.csv`

**Usage**:
```bash
python3 scripts/01_data_collection_fetch_historical_games.py
```

**Time**: ~20-40 minutes (due to API rate limiting)

---

#### `02_data_collection_add_injuries.py`
**Purpose**: Adds injury features to existing game dataset  
**What it does**:
- Fetches injury data from ESPN API for each game
- Calculates injury severity metrics
- Adds injury features to dataset
- Saves to `data/final_dataset_with_injuries.csv`

**Usage**:
```bash
python3 scripts/02_data_collection_add_injuries.py
```

**Time**: ~15-30 minutes (due to API rate limiting)

**Note**: Requires `data/final_dataset_raw_games.csv` (or `data/nba_games.csv` for backward compatibility)

---

## Data Files

### Final Dataset Files

#### `data/final_dataset_with_injuries.csv`
**Status**: ✅ **FINAL DATASET - USE THIS FOR TRAINING**  
**Contents**:
- All historical games
- DraftKings opening and closing spreads
- Injury features (players out, severity, etc.)
- Ready for model training

#### `data/final_dataset_raw_games.csv`
**Status**: Intermediate file  
**Contents**:
- Historical games with spreads
- No injury features yet
- Created by `01_data_collection_fetch_historical_games.py`

---

## Backward Compatibility

The scripts maintain backward compatibility with old file names:
- `nba_games.csv` → Will be used if `final_dataset_raw_games.csv` not found
- `nba_games_with_injuries.csv` → Will be used if `final_dataset_with_injuries.csv` not found

However, **new data collection will use the new naming convention**.

---

## Quick Start

### First Time Setup (Full Data Collection)
```bash
# Run complete pipeline
python3 scripts/00_data_collection_run_all.py
```

### Update Existing Data
```bash
# Just fetch new games
python3 scripts/01_data_collection_fetch_historical_games.py

# Then add injuries
python3 scripts/02_data_collection_add_injuries.py
```

---

## Other Scripts

### Utility Scripts
- `check_spreads.py` - Verify spread data quality
- `backfill_historical_spreads.py` - Backfill missing spreads
- `fetch_historical_spreads.py` - Legacy spread fetching

### Prediction Scripts
- `predict_upcoming_games.py` - Make predictions for upcoming games
- `run_full_pipeline.py` - Train model and evaluate
- `daily_predictions.py` - Daily automated predictions

---

## File Naming Convention

**Data Collection Scripts**: `XX_data_collection_*.py`
- `00_` = Master/orchestration script
- `01_` = Step 1 (fetch games)
- `02_` = Step 2 (add injuries)

**Data Files**: `final_dataset_*.csv`
- `final_dataset_raw_games.csv` = Games with spreads (intermediate)
- `final_dataset_with_injuries.csv` = Complete dataset (final)

---

## Notes

- All scripts check for new file names first, then fallback to old names
- The final dataset (`final_dataset_with_injuries.csv`) is what the model uses for training
- Data collection scripts are rate-limited to respect ESPN API limits

