#!/usr/bin/env python3
"""
Comprehensive model diagnostics:
- Multicollinearity analysis
- Feature correlation analysis
- Feature importance analysis
- Model improvement suggestions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, confusion_matrix
import lightgbm as lgb
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

from run_full_pipeline import load_game_data, build_features, prepare_model_data

print("="*70)
print("COMPREHENSIVE MODEL DIAGNOSTICS")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
games_df = load_game_data()
if games_df is None or len(games_df) == 0:
    print("❌ Failed to load data")
    sys.exit(1)

print(f"✓ Loaded {len(games_df)} games")

# Build features
print("\n[2/5] Building features...")
features_df = build_features(games_df, include_spread=True)
print(f"✓ Built features for {len(features_df)} games")

# Prepare model data
print("\n[3/5] Preparing model data...")
X, y, feature_names, metadata = prepare_model_data(features_df, target='residual')
print(f"✓ Prepared {len(feature_names)} features")

# ============================================================================
# MULTICOLLINEARITY ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("MULTICOLLINEARITY ANALYSIS")
print("="*70)

# Create feature DataFrame
feature_df = pd.DataFrame(X, columns=feature_names)

# Calculate correlation matrix
corr_matrix = feature_df.corr().abs()

# Find highly correlated feature pairs
high_corr_pairs = []
threshold = 0.8  # Correlation threshold

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        if corr_value >= threshold:
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_value
            })

if high_corr_pairs:
    print(f"\n⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (r >= {threshold}):")
    print("-"*70)
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
    for idx, row in high_corr_df.iterrows():
        print(f"  {row['feature1']:<30} <-> {row['feature2']:<30} r={row['correlation']:.3f}")
    
    # Suggest which features to remove (keep the one with higher variance or importance)
    print(f"\n💡 Recommendation: Consider removing one feature from each pair")
    print("   (Keep the feature with higher variance or importance)")
else:
    print(f"\n✓ No highly correlated features found (threshold: {threshold})")

# Check for perfect correlations (r = 1.0)
perfect_corr = [p for p in high_corr_pairs if p['correlation'] >= 0.99]
if perfect_corr:
    print(f"\n⚠️  Found {len(perfect_corr)} perfectly correlated pairs (r >= 0.99):")
    for pair in perfect_corr:
        print(f"  {pair['feature1']} <-> {pair['feature2']} (r={pair['correlation']:.3f})")
    print("  These should definitely be removed!")

# ============================================================================
# FEATURE CORRELATION WITH TARGET
# ============================================================================
print("\n" + "="*70)
print("FEATURE-TARGET CORRELATION")
print("="*70)

target_correlations = []
for col in feature_names:
    col_idx = feature_names.index(col)
    corr = np.corrcoef(feature_df[col], y)[0, 1]
    target_correlations.append({
        'feature': col,
        'correlation': abs(corr),
        'direction': 'positive' if corr > 0 else 'negative'
    })

target_corr_df = pd.DataFrame(target_correlations).sort_values('correlation', ascending=False)
print(f"\nTop 10 features most correlated with target (residual):")
print("-"*70)
for idx, row in target_corr_df.head(10).iterrows():
    print(f"  {row['feature']:<40} r={row['correlation']:.3f} ({row['direction']})")

print(f"\nBottom 10 features (least correlated):")
print("-"*70)
for idx, row in target_corr_df.tail(10).iterrows():
    print(f"  {row['feature']:<40} r={row['correlation']:.3f}")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Train a quick model to get feature importance
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
y_train = y[:split_idx]

dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
params = {
    "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
    "learning_rate": 0.03, "num_leaves": 20, "feature_fraction": 0.7,
    "bagging_fraction": 0.7, "bagging_freq": 5, "min_child_samples": 30,
    "lambda_l1": 0.1, "lambda_l2": 0.1, "max_depth": 6,
    "min_split_gain": 0.1, "verbose": -1, "seed": 42
}

model = lgb.train(params, dtrain, num_boost_round=200, callbacks=[lgb.log_evaluation(period=0)])
feature_importance = model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 15 most important features:")
print("-"*70)
for idx, row in importance_df.head(15).iterrows():
    marker = "🏥" if any(x in row['feature'].lower() for x in ['injury', 'players_out']) else "  "
    print(f"{marker} {row['feature']:<40} {row['importance']:>10.0f}")

print(f"\nBottom 10 least important features:")
print("-"*70)
for idx, row in importance_df.tail(10).iterrows():
    print(f"  {row['feature']:<40} {row['importance']:>10.0f}")

# ============================================================================
# FEATURE REMOVAL RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("FEATURE REMOVAL RECOMMENDATIONS")
print("="*70)

features_to_remove = []

# 1. Remove low-importance features (bottom 5%)
low_importance_threshold = importance_df['importance'].quantile(0.05)
low_importance = importance_df[importance_df['importance'] <= low_importance_threshold]
if len(low_importance) > 0:
    print(f"\n1. Low Importance Features (bottom 5%, threshold: {low_importance_threshold:.0f}):")
    for idx, row in low_importance.iterrows():
        print(f"   - {row['feature']} (importance: {row['importance']:.0f})")
        features_to_remove.append(row['feature'])

# 2. Remove one from each highly correlated pair
if high_corr_pairs:
    print(f"\n2. Highly Correlated Features (remove one from each pair):")
    removed_from_pairs = set()
    for pair in high_corr_pairs:
        if pair['feature1'] not in removed_from_pairs and pair['feature2'] not in removed_from_pairs:
            # Remove the one with lower importance
            imp1 = importance_df[importance_df['feature'] == pair['feature1']]['importance'].values[0]
            imp2 = importance_df[importance_df['feature'] == pair['feature2']]['importance'].values[0]
            to_remove = pair['feature1'] if imp1 < imp2 else pair['feature2']
            to_keep = pair['feature2'] if imp1 < imp2 else pair['feature1']
            print(f"   - Remove: {to_remove} (keep: {to_keep}, r={pair['correlation']:.3f})")
            features_to_remove.append(to_remove)
            removed_from_pairs.add(to_remove)

# 3. Remove features with very low target correlation
low_target_corr = target_corr_df[target_corr_df['correlation'] < 0.05]
if len(low_target_corr) > 0:
    print(f"\n3. Low Target Correlation (r < 0.05):")
    for idx, row in low_target_corr.head(5).iterrows():
        if row['feature'] not in features_to_remove:
            print(f"   - {row['feature']} (r={row['correlation']:.3f})")
            features_to_remove.append(row['feature'])

features_to_remove = list(set(features_to_remove))  # Remove duplicates

if features_to_remove:
    print(f"\n📋 Summary: {len(features_to_remove)} features recommended for removal:")
    for feat in features_to_remove:
        print(f"   - {feat}")
else:
    print("\n✓ No features recommended for removal")

# ============================================================================
# TEST MODEL WITH FEATURE REMOVAL
# ============================================================================
if features_to_remove:
    print("\n" + "="*70)
    print("TESTING MODEL WITH FEATURE REMOVAL")
    print("="*70)
    
    # Remove features
    features_to_keep = [f for f in feature_names if f not in features_to_remove]
    feature_indices = [feature_names.index(f) for f in features_to_keep]
    X_reduced = X[:, feature_indices]
    
    print(f"\n  Original features: {len(feature_names)}")
    print(f"  Reduced features: {len(features_to_keep)}")
    print(f"  Removed: {len(features_to_remove)} features")
    
    # Train/test split
    split_idx = int(len(X_reduced) * 0.8)
    X_train = X_reduced[:split_idx]
    X_test = X_reduced[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    metadata_test = metadata.iloc[split_idx:].copy()
    
    # Train model
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features_to_keep)
    model_reduced = lgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    
    # Evaluate
    train_pred = model_reduced.predict(X_train)
    test_pred = model_reduced.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # Cover predictions
    test_pred_cover = (test_pred > 0.40).astype(int)
    test_true_cover = (metadata_test['residual'].values > 0).astype(int)
    test_acc = (test_pred_cover == test_true_cover).mean()
    
    print(f"\n  Results with reduced features:")
    print(f"    Train MAE: {train_mae:.3f}")
    print(f"    Test MAE:  {test_mae:.3f}")
    print(f"    Test Accuracy: {test_acc:.1%}")
    print(f"    Home Cover Rate: {test_pred_cover.mean():.1%}")

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "="*70)
print("THRESHOLD OPTIMIZATION")
print("="*70)

split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]
metadata_test = metadata.iloc[split_idx:].copy()

dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
model = lgb.train(params, dtrain, num_boost_round=200, callbacks=[lgb.log_evaluation(period=0)])

test_pred = model.predict(X_test)
test_true_cover = (metadata_test['residual'].values > 0).astype(int)

# Test different thresholds
thresholds = np.arange(-2, 2, 0.1)
best_threshold = 0
best_acc = 0
results = []

for threshold in thresholds:
    pred_cover = (test_pred > threshold).astype(int)
    acc = (pred_cover == test_true_cover).mean()
    home_rate = pred_cover.mean()
    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'home_cover_rate': home_rate
    })
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

results_df = pd.DataFrame(results)
print(f"\n  Best threshold: {best_threshold:.2f}")
print(f"  Best accuracy: {best_acc:.1%}")
print(f"  Home cover rate at best threshold: {results_df[results_df['threshold'] == best_threshold]['home_cover_rate'].values[0]:.1%}")
print(f"  Actual home cover rate: {test_true_cover.mean():.1%}")

# Show top 5 thresholds
print(f"\n  Top 5 thresholds by accuracy:")
print("-"*70)
for idx, row in results_df.nlargest(5, 'accuracy').iterrows():
    print(f"    Threshold: {row['threshold']:>5.2f} | Accuracy: {row['accuracy']:.1%} | Home Rate: {row['home_cover_rate']:.1%}")

print("\n" + "="*70)
print("✓ Diagnostics complete!")
print("="*70)

