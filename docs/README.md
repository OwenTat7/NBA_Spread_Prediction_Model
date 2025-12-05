# NBA Spread Prediction System

A production-ready machine learning pipeline for predicting NBA point spreads and ATS (Against The Spread) outcomes.

## 🎯 Project Overview

This system predicts NBA game margins and betting outcomes using:
- Historical game data from ESPN APIs
- Advanced feature engineering (ELO ratings, rolling stats, rest days)
- Machine learning models (LightGBM, Ridge regression)
- Time-series validation and backtesting

## 📁 Project Structure

```
.
├── nba_data_collection.ipynb      # Data collection notebook
├── nba_spread_prediction.ipynb    # Main prediction pipeline ⭐
├── fetch_historical_data.py       # Standalone data collection script
├── fetch_historical_data_test.py  # Quick test script
├── test_prediction_pipeline.py    # Data validation script
├── nba_games_test.csv              # Test dataset (213 games)
├── nba_games.csv                   # Full dataset (when complete)
├── STATUS.md                       # Current project status
├── PROGRESS.md                     # Detailed progress report
└── README.md                       # This file
```

## 🚀 Quick Start

### Step 1: Test the Pipeline (Ready Now!)

1. Open `nba_spread_prediction.ipynb` in Jupyter
2. Run all cells sequentially
3. The notebook will:
   - Load test data (213 games)
   - Build features (ELO, rolling stats, etc.)
   - Train models (Baseline, Ridge, LightGBM)
   - Generate predictions and evaluate performance
   - Save models to `models/` directory

### Step 2: Full Dataset (In Progress)

The full historical data collection is running in the background. Once `nba_games.csv` is created:
- Re-run the prediction notebook
- It will automatically use the full dataset (~1,200+ games)
- Better model performance expected

## 📊 What You'll Get

### Models
- **Baseline**: Closing spread (market baseline)
- **Ridge Regression**: Linear model with regularization
- **LightGBM**: Gradient boosting (main model)

### Outputs
- Feature importance rankings
- Model performance metrics (MAE, RMSE, R²)
- Betting metrics (cover accuracy, average edge)
- Backtest results
- Saved models for production use

### Files Generated
- `models/lgb_spread_model.txt` - Trained LightGBM model
- `models/feature_names.txt` - Feature list
- `models/feature_importance.csv` - Feature rankings
- `data/processed/features_with_targets.csv` - Full feature dataset
- `data/processed/backtest_predictions.csv` - Backtest results

## 🔧 Features

### Feature Engineering
- **ELO Ratings**: Team strength ratings updated after each game
- **Rolling Statistics**: Last 3/5/10 games (margin, points for/against, consistency)
- **Rest Days**: Days between games, back-to-back detection
- **Temporal**: Day of week, month, weekend indicators
- **Interactions**: ELO × rest, form × ELO, etc.

### Models
- Time-series cross-validation (no data leakage)
- Walk-forward backtesting
- SHAP interpretability
- Model drift detection

## 📈 Current Status

✅ **Ready to Use**: Test dataset (213 games) available  
🔄 **In Progress**: Full historical data collection  
✅ **Complete**: Prediction pipeline fully implemented

## 📝 Requirements

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn shap joblib requests
```

See `nba_spread_prediction.ipynb` for full requirements list.

## 🎓 How It Works

1. **Data Collection**: Fetches game data from ESPN APIs
2. **Feature Engineering**: Builds predictive features from historical games
3. **Model Training**: Trains multiple models with time-series validation
4. **Evaluation**: Tests on held-out data with betting-relevant metrics
5. **Prediction**: Generates predictions for future games

## ⚠️ Important Notes

- **Closing Spreads**: Currently estimated from ELO. For production, integrate actual betting line data.
- **Data Quality**: Accurate injury/line data significantly improves predictions
- **Sample Size**: Early season predictions have higher variance
- **Legal**: Be aware of local gambling regulations

## 📚 Documentation

- `STATUS.md` - Current project status
- `PROGRESS.md` - Detailed progress report
- `espn-api-docs.md` - ESPN API documentation

## 🔍 Monitoring

Check data collection progress:
```bash
# Check if process is running
ps aux | grep fetch_historical_data

# View log
tail -f fetch_log.txt

# Check output
ls -lh nba_games.csv
```

## 🎯 Next Steps

1. ✅ Test pipeline with current data (213 games)
2. ⏳ Wait for full dataset collection
3. 🔄 Re-run with full dataset
4. 📊 Analyze results and feature importance
5. 🚀 Deploy for production predictions

---

**Ready to start?** Open `nba_spread_prediction.ipynb` and run all cells!

