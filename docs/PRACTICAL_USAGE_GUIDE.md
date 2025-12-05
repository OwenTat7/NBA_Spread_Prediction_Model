# Practical Usage Guide - Making Real Predictions

This guide shows you how to use the model to predict actual NBA games and export results.

## 🚀 Quick Start

### Option 1: Predict Upcoming Games (Recommended)

```bash
python scripts/predict_upcoming_games.py
```

This will:
- ✅ Fetch upcoming games from ESPN API (next 7 days)
- ✅ Train model on historical data
- ✅ Generate predictions for each game
- ✅ Export to Excel with formatted output
- ✅ Export to CSV for easy sharing

**Output files:**
- `predictions_YYYYMMDD_HHMMSS.xlsx` - Formatted Excel file
- `predictions_YYYYMMDD_HHMMSS.csv` - CSV file

### Option 2: Daily Predictions

```bash
python scripts/daily_predictions.py
```

Run this daily to get predictions for today's games.

---

## 📊 Understanding the Output

### Excel File Columns

| Column | Description |
|--------|-------------|
| **Date** | Game date |
| **Time** | Game time |
| **Away Team** | Visiting team |
| **Home Team** | Home team |
| **Spread** | Predicted spread (from ELO) |
| **Predicted Margin** | Predicted final margin (home - away) |
| **Predicted Residual** | Model's edge (positive = home covers, negative = away covers) |
| **Cover Prediction** | Home Covers / Away Covers |
| **Confidence** | High / Medium / Low |
| **Recommendation** | Betting recommendation |

### Color Coding

- 🟢 **Green** = Home team predicted to cover
- 🔴 **Red** = Away team predicted to cover

### Confidence Levels

- **High**: Residual > 3 points (strong edge)
- **Medium**: Residual 1.5-3 points (moderate edge)
- **Low**: Residual < 1.5 points (weak edge)

---

## 💡 Practical Use Cases

### 1. Daily Betting Decisions

**Morning Routine:**
```bash
# Run predictions
python scripts/daily_predictions.py

# Open Excel file
open predictions_*.xlsx
```

**How to Use:**
1. Check games with "High" or "Medium" confidence
2. Focus on recommendations marked "Strong"
3. Compare predicted residual vs. actual betting line
4. Make informed betting decisions

### 2. Weekly Planning

**Sunday Night:**
```bash
# Get predictions for upcoming week
python scripts/predict_upcoming_games.py
```

**Plan Your Week:**
- Review all games for the week
- Identify high-confidence games
- Track line movements
- Set betting budget

### 3. Research & Analysis

**For Analysis:**
- Export to CSV for further analysis
- Compare predictions vs. actual results
- Track model performance over time
- Build your own analysis on top

### 4. Sharing with Friends

**Easy Sharing:**
- Excel file is ready to email/share
- CSV can be imported into Google Sheets
- Formatted and easy to read

---

## 📈 Interpreting Predictions

### Example Output

```
Date         Matchup                    Spread   Pred Margin   Cover           Confidence
12/02 07:30  LAL @ BOS                  -5.0       -2.3        Away Covers     Medium
12/02 08:00  GSW @ DEN                  -3.5        2.1        Home Covers     Low
12/03 07:00  PHX @ MIA                  +1.5        5.2        Home Covers     High
```

### What This Means

**Game 1: LAL @ BOS**
- Spread: -5.0 (Boston favored by 5)
- Predicted margin: -2.3 (Boston wins by 2.3)
- **Prediction**: Lakers cover (they lose by less than 5)
- **Confidence**: Medium (residual = 2.7)

**Game 3: PHX @ MIA**
- Spread: +1.5 (Phoenix underdog by 1.5)
- Predicted margin: 5.2 (Miami wins by 5.2)
- **Prediction**: Miami covers (wins by more than 1.5)
- **Confidence**: High (residual = 3.7)

---

## 🎯 Best Practices

### 1. Combine with Line Shopping

- Get predictions from model
- Check multiple sportsbooks for best lines
- Look for line movement
- Bet when model edge aligns with good line

### 2. Bankroll Management

- Model is ~56% accurate (not 100%)
- Only bet games with Medium+ confidence
- Size bets based on confidence
- Never bet more than you can afford to lose

### 3. Track Performance

- Keep a log of predictions vs. results
- Track which confidence levels perform best
- Adjust strategy based on results
- Remember: variance is high in sports betting

### 4. Compare to Market

- Compare predicted spread to actual betting line
- Large differences may indicate value
- Small differences mean market agrees with model
- Use model as one input, not the only input

---

## 🔧 Customization

### Change Number of Days

Edit `predict_upcoming_games.py`:

```python
# Change days_ahead parameter
upcoming_df = fetch_upcoming_games(days_ahead=14)  # 2 weeks
```

### Customize Output

Edit `export_to_excel()` function to:
- Add more columns
- Change formatting
- Add charts/graphs
- Customize colors

### Filter Games

Add filters in `main()`:

```python
# Only high confidence games
predictions_df = predictions_df[predictions_df['confidence'] == 'High']

# Only games with strong recommendations
predictions_df = predictions_df[predictions_df['recommendation'].str.contains('Strong')]
```

---

## 📱 Automation Ideas

### Daily Email Report

```bash
# Add to crontab (Linux/Mac)
0 8 * * * cd /path/to/project && python scripts/daily_predictions.py && mail -s "Daily NBA Predictions" your@email.com < predictions_*.csv
```

### Slack/Discord Bot

- Run predictions script
- Parse CSV/Excel output
- Post to Slack/Discord channel
- Format as nice message

### Google Sheets Integration

1. Export to CSV
2. Import to Google Sheets
3. Set up scheduled import
4. Share with team/friends

---

## ⚠️ Important Reminders

1. **Model Accuracy**: ~56% cover accuracy (not guaranteed wins)
2. **Variance**: Sports betting has high variance
3. **Bankroll**: Only bet what you can afford to lose
4. **Line Shopping**: Always check multiple books
5. **Track Results**: Monitor performance over time

---

## 🐛 Troubleshooting

### "No upcoming games found"

- Check your internet connection
- ESPN API may be down
- Try running again later
- Check if NBA season is active

### "Error training model"

- Ensure `data/nba_games.csv` exists
- Check file has required columns
- Verify Python packages installed

### "Excel file won't open"

- Ensure `openpyxl` is installed: `pip install openpyxl`
- Try opening CSV file instead
- Check file isn't corrupted

---

## 📊 Expected Performance

### Cross-Validated Performance (Realistic Expectation) ⭐

Based on historical backtesting with time-series cross-validation:
- **Cover Accuracy**: **~56.1%** (cross-validated on held-out data)
- **Better than**: Random (50%) and Baseline (43.6%)
- **MAE**: ~11.3 points
- **Best Use**: Games with Medium+ confidence

### Training Set Performance (Optimistic)

- **Cover Accuracy**: ~65.0% (evaluated on training data)
- **Note**: This is higher because the model saw this data during training
- **For Real Predictions**: Expect closer to 56.1% (cross-validated)

### Understanding the Difference

- **65.0%** = Training accuracy (model evaluated on data it was trained on)
- **56.1%** = Cross-validated accuracy (model evaluated on unseen data) ⭐ **Use this for expectations**

**Why the difference?**
- Training accuracy can be optimistic (model may overfit to training patterns)
- Cross-validation tests on held-out data (more realistic for new games)
- **56.1% is the more accurate expectation** for real-world predictions

**Remember**: Past performance doesn't guarantee future results!

---

## 🔗 Related Files

- `scripts/predict_upcoming_games.py` - Main prediction script
- `scripts/daily_predictions.py` - Daily automation script
- `scripts/run_full_pipeline.py` - Model training pipeline
- `docs/ACCURACY_REPORT.md` - Model performance details

---

**Happy Predicting!** 🏀📊

