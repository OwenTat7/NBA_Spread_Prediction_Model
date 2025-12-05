# Accuracy Metrics Clarification

## ⚠️ Important: Two Different Accuracy Numbers

You may see two different accuracy numbers in the documentation. Here's why:

### 65.0% - Training Set Accuracy (Optimistic)
- **What it is**: Model performance on the full training dataset
- **Why it's higher**: The model was trained on this data, so it can "remember" patterns
- **Where you see it**: `ACCURACY_REPORT.md`, `PERFORMANCE_COMPARISON.md`
- **Use case**: Shows model learned patterns, but not realistic for new games

### 56.1% - Cross-Validated Accuracy (Realistic) ⭐
- **What it is**: Model performance on held-out data using time-series cross-validation
- **Why it's lower**: Tests on data the model hasn't seen (more realistic)
- **Where you see it**: `INJURY_FEATURE_ANALYSIS_SUMMARY.md`, `PRACTICAL_USAGE_GUIDE.md`
- **Use case**: **Use this for real-world expectations**

## 📊 Quick Reference

| Metric Type | Accuracy | When to Use |
|------------|----------|-------------|
| **Training Set** | 65.0% | Shows model learned, but optimistic |
| **Cross-Validated** ⭐ | **56.1%** | **Realistic expectation for new games** |

## 🎯 Which Number to Use?

### For Real Predictions: **56.1%**

When using the model to predict upcoming games, expect:
- **~56% cover accuracy** (not 65%)
- This is still **better than random (50%)** and **better than baseline (43.6%)**
- Focus on games with Medium+ confidence for best results

### Why Both Numbers Exist

1. **Training Set (65%)**: 
   - Shows the model successfully learned patterns
   - Useful for comparing model configurations
   - Can be misleading for expectations

2. **Cross-Validated (56.1%)**:
   - Tests generalization to new data
   - More realistic for production use
   - Better estimate of real-world performance

## 📈 What This Means

- **Training**: Model achieves 65% on data it saw
- **Real World**: Expect ~56% on new games
- **Still Good**: 56% is 6 percentage points better than random (50%)
- **Better than Baseline**: 12.5 percentage points better than baseline (43.6%)

## 🔍 How Cross-Validation Works

Time-series cross-validation:
1. Split data chronologically into 3 folds
2. Train on earlier data
3. Test on later data (the model hasn't seen)
4. Average results across folds

This prevents:
- Data leakage (model seeing future data)
- Overfitting to training patterns
- Unrealistic accuracy estimates

## ✅ Summary

- **65.0%** = Training accuracy (what you see in some reports)
- **56.1%** = Cross-validated accuracy (what to expect) ⭐
- **Use 56.1%** when setting expectations for real predictions
- **Still profitable** if used correctly (56% > 50% baseline)

