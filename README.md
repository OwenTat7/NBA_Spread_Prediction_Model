# NBA Spread Prediction Model

A machine learning model for predicting NBA game spread outcomes using LightGBM, incorporating ELO ratings, rolling statistics, injury data, and DraftKings spreads.

## 🎯 Model Performance

- **Test Accuracy**: 61.9% (vs 50% baseline)
- **Features**: 43 features including ELO, rolling stats, injuries, and market data
- **Generalization**: Excellent (1.9% train/test gap)

## 📁 Project Structure

```
NBA_Prediction_Model/
├── data/                          # Training datasets
│   ├── final_dataset_raw_games.csv
│   └── final_dataset_with_injuries.csv
├── predictions/                   # Prediction outputs
│   ├── predictions_latest.xlsx    # Latest predictions (overwritten each run)
│   ├── predictions_latest.csv
│   ├── predictions_history.xlsx   # Full prediction history
│   ├── predictions_history.csv
│   ├── predictions_summary.xlsx   # Combined history + correctness
│   └── prediction_correctness.xlsx
├── scripts/                      # Main scripts
│   ├── 00_data_collection_run_all.py
│   ├── 01_data_collection_fetch_historical_games.py
│   ├── 02_data_collection_add_injuries.py
│   ├── daily_predictions.py       # Daily automation script
│   ├── predict_upcoming_games.py  # Generate predictions
│   ├── run_full_pipeline.py       # Train and evaluate model
│   ├── injury_features.py         # Injury feature helpers
│   └── model_diagnostics.py       # Model analysis tools
├── docs/                         # Documentation
│   ├── PRACTICAL_USAGE_GUIDE.md   # Usage guide
│   └── QUICK_START.md            # Quick start guide
├── index.qmd                     # Comprehensive project report (Quarto)
└── logs/                         # Execution logs
```

## 🚀 Quick Start

### 1. Collect Data
```bash
python3 scripts/00_data_collection_run_all.py --non-interactive
```

### 2. Train Model
```bash
python3 scripts/run_full_pipeline.py
```

### 3. Generate Predictions
```bash
python3 scripts/predict_upcoming_games.py
```

### 4. Daily Automation
```bash
python3 scripts/daily_predictions.py
```

## 📊 Output Files

All predictions are saved in the `predictions/` folder:

- **predictions_latest.xlsx**: Latest predictions (next 2 days)
- **predictions_history.xlsx**: Complete prediction history
- **predictions_summary.xlsx**: Combined workbook with History and Correctness sheets

## 🔧 Key Features

- **ELO Rating System**: Dynamic team strength ratings
- **Rolling Statistics**: 3, 5, and 10-game averages for margins and points
- **Injury Data**: Severity-weighted injury features
- **DraftKings Spreads**: Market-based features (opening/closing spreads, line movement)
- **Time-Series Validation**: Proper train/test split for time-series data

## 📖 Documentation

- **index.qmd**: Complete project documentation (render with Quarto to create webpage)
- **docs/PRACTICAL_USAGE_GUIDE.md**: Detailed usage instructions
- **docs/QUICK_START.md**: Quick reference guide

## 🔄 Daily Automation

Set up a cron job to run predictions daily:
```bash
# Edit crontab
crontab -e

# Add line (example: run at 1:30 PM daily)
30 13 * * * cd "/path/to/NBA_Prediction_Model" && /opt/anaconda3/bin/python3 scripts/daily_predictions.py >> logs/daily_predictions.log 2>&1
```

## 📝 Requirements

- Python 3.8+
- pandas
- lightgbm
- openpyxl
- tqdm
- requests

Install with:
```bash
pip install pandas lightgbm openpyxl tqdm requests
```

## 🎓 Model Details

The model predicts game residuals (actual margin - closing spread) using:
- 43 engineered features
- LightGBM gradient boosting
- Regularization to prevent overfitting
- Optimal threshold tuning for balanced predictions

See `index.qmd` (render with Quarto) for complete methodology and results.
