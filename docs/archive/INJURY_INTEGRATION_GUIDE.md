# Injury Features Integration Guide

## ✅ What's Been Built

### 1. Injury Feature Module (`injury_features.py`)
- Extracts injury data from ESPN API game summaries
- Calculates injury severity scores
- Creates 13 injury-related features:
  - `players_out_home/away` - Count of players out
  - `players_dtd_home/away` - Day-to-day players
  - `injury_severity_home/away` - Weighted severity score
  - `star_out_home/away` - Binary: significant player out
  - `total_injury_impact_home/away` - Combined impact score
  - `injury_severity_diff` - Home vs away difference
  - `players_out_diff` - Count difference
  - `total_injury_impact_diff` - Impact difference

### 2. Integration Script (`add_injuries_to_pipeline.py`)
- Fetches injury data for existing games
- Analyzes multicollinearity
- Reports correlations with spread features

### 3. Updated Prediction Pipeline
- `build_features()` now accepts `include_injuries` parameter
- Automatically detects injury features in dataset
- Creates interaction features (injury × elo, injury beyond spread)
- Includes multicollinearity checks

## 🚀 How to Use

### Step 1: Add Injury Data to Your Dataset

**Option A: Quick Test (50 games)**
```bash
python add_injuries_to_pipeline.py
# When prompted, it will test with 50 games first
```

**Option B: Full Dataset**
```bash
# Edit add_injuries_to_pipeline.py to skip the test
# Or run directly on your dataset
python -c "
from injury_features import add_injury_features_to_games
import pandas as pd
df = pd.read_csv('nba_games.csv')
df_with_injuries = add_injury_features_to_games(df)
df_with_injuries.to_csv('nba_games_with_injuries.csv', index=False)
"
```

**Note**: Full dataset (3,160 games) will take ~30-45 minutes due to API rate limiting.

### Step 2: Run Prediction Pipeline

The notebook will automatically detect injury features:

```python
# In nba_spread_prediction.ipynb
games_df = load_game_data('nba_games_with_injuries.csv')  # Or auto-detect
features_df = build_features(games_df, include_injuries=True)
```

### Step 3: Review Multicollinearity

The notebook will automatically:
- Calculate correlations between injury and spread features
- Warn if correlations > 0.8
- Show correlation matrix

## 📊 Multicollinearity Handling

### Built-in Safeguards

1. **Correlation Analysis**: Automatically calculated and reported
2. **Feature Importance**: LightGBM will show if injuries add value
3. **Regularization**: Ridge regression handles multicollinearity
4. **Tree Models**: LightGBM is less sensitive to multicollinearity

### If High Correlation Detected

**Option 1: Remove Redundant Features**
```python
# In build_features(), exclude high-correlation features
if corr > 0.8:
    df = df.drop(columns=['redundant_injury_feature'])
```

**Option 2: Create Interaction Features**
```python
# Instead of raw injury counts, use:
df['injury_beyond_spread'] = df['injury_severity_diff'] - (df['spread_move'] * 0.5)
# This captures injury impact NOT in spread
```

**Option 3: Feature Selection**
```python
# Use feature importance to drop low-importance features
# LightGBM will naturally down-weight redundant features
```

## 🧪 Testing Strategy

### Compare Models With/Without Injuries

```python
# Model 1: Without injuries
features_no_inj = build_features(games_df, include_injuries=False)
model_no_inj = train_lightgbm(...)

# Model 2: With injuries
features_with_inj = build_features(games_df, include_injuries=True)
model_with_inj = train_lightgbm(...)

# Compare accuracy
# If injuries improve accuracy, keep them
# If not, they're redundant
```

## 📈 Expected Results

### Best Case
- **Correlation**: Low (r < 0.5)
- **Accuracy Improvement**: +5-10%
- **Cover Accuracy**: 67-70% (up from 65%)

### Worst Case
- **Correlation**: High (r > 0.8)
- **Accuracy**: No improvement
- **Action**: Remove redundant features

### Most Likely
- **Correlation**: Moderate (r = 0.4-0.7)
- **Accuracy Improvement**: +2-5%
- **Cover Accuracy**: 66-68%

## 🔍 Monitoring

### Check Feature Importance

After training, review:
```python
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

# If injury features are in bottom 20%, consider removing
```

### Check Correlations

```python
from injury_features import analyze_injury_correlations
corr_matrix = analyze_injury_correlations(features_df)
print(corr_matrix)
```

## ⚠️ Important Notes

1. **API Rate Limiting**: Injury data fetching is slow (0.5s delay per game)
2. **Historical Data**: Injury data only available for games with summaries
3. **Missing Data**: Some games may have no injury data (defaults to 0)
4. **Real-Time**: For live predictions, fetch injury data daily before games

## 🎯 Next Steps

1. ✅ Run `add_injuries_to_pipeline.py` on your dataset
2. ✅ Re-run prediction pipeline with injury features
3. ✅ Review correlation matrix and feature importance
4. ✅ Compare accuracy with/without injuries
5. ✅ Keep or remove based on results

---

**Questions?** Check the correlation matrix and feature importance - they'll tell you if injuries add value beyond spreads!


