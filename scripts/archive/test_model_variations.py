#!/usr/bin/env python3
"""
Test different model configurations to find the best setup
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, confusion_matrix
import lightgbm as lgb
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

from run_full_pipeline import load_game_data, build_features, prepare_model_data

print("="*70)
print("TESTING MODEL VARIATIONS")
print("="*70)

# Load and prepare data
games_df = load_game_data()
features_df = build_features(games_df, include_spread=True)
X, y, feature_names, metadata = prepare_model_data(features_df, target='residual')

# Train/test split
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]
metadata_test = metadata.iloc[split_idx:].copy()
test_true_cover = (metadata_test['residual'].values > 0).astype(int)

print(f"\nTraining on {len(X_train)} games, testing on {len(X_test)} games")
print(f"Features: {len(feature_names)}")

# Test different configurations
configs = [
    {
        'name': 'Current (Regularized)',
        'params': {
            "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
            "learning_rate": 0.03, "num_leaves": 20, "feature_fraction": 0.7,
            "bagging_fraction": 0.7, "bagging_freq": 5, "min_child_samples": 30,
            "lambda_l1": 0.1, "lambda_l2": 0.1, "max_depth": 6,
            "min_split_gain": 0.1, "verbose": -1, "seed": 42
        }
    },
    {
        'name': 'More Regularized',
        'params': {
            "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
            "learning_rate": 0.02, "num_leaves": 15, "feature_fraction": 0.6,
            "bagging_fraction": 0.6, "bagging_freq": 5, "min_child_samples": 40,
            "lambda_l1": 0.2, "lambda_l2": 0.2, "max_depth": 5,
            "min_split_gain": 0.2, "verbose": -1, "seed": 42
        }
    },
    {
        'name': 'Less Regularized',
        'params': {
            "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
            "learning_rate": 0.05, "num_leaves": 25, "feature_fraction": 0.75,
            "bagging_fraction": 0.75, "bagging_freq": 5, "min_child_samples": 25,
            "lambda_l1": 0.05, "lambda_l2": 0.05, "max_depth": 7,
            "min_split_gain": 0.05, "verbose": -1, "seed": 42
        }
    }
]

results = []

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"{'='*70}")
    
    # Train model
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_test, label=y_test, reference=dtrain, feature_name=feature_names)
    model = lgb.train(config['params'], dtrain, num_boost_round=300, 
                     valid_sets=[dval],
                     callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), 
                               lgb.log_evaluation(period=0)])
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # Find optimal threshold
    thresholds = np.arange(-2, 3, 0.1)
    best_threshold = 0
    best_acc = 0
    
    for threshold in thresholds:
        pred_cover = (test_pred > threshold).astype(int)
        acc = (pred_cover == test_true_cover).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    # Metrics at optimal threshold
    optimal_pred_cover = (test_pred > best_threshold).astype(int)
    cm = confusion_matrix(test_true_cover, optimal_pred_cover)
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[0,0]) if (cm[1,1] + cm[0,0]) > 0 else 0
    home_rate = optimal_pred_cover.mean()
    
    results.append({
        'name': config['name'],
        'train_mae': train_mae,
        'test_mae': test_mae,
        'mae_diff': train_mae - test_mae,
        'best_threshold': best_threshold,
        'test_accuracy': best_acc,
        'precision': precision,
        'recall': recall,
        'home_cover_rate': home_rate
    })
    
    print(f"  Train MAE: {train_mae:.3f}")
    print(f"  Test MAE:  {test_mae:.3f}")
    print(f"  MAE Diff:  {train_mae - test_mae:.3f}")
    print(f"  Optimal Threshold: {best_threshold:.2f}")
    print(f"  Test Accuracy: {best_acc:.1%}")
    print(f"  Home Cover Rate: {home_rate:.1%} (actual: {test_true_cover.mean():.1%})")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY - BEST CONFIGURATION")
print(f"{'='*70}")

results_df = pd.DataFrame(results)
best_config = results_df.loc[results_df['test_accuracy'].idxmax()]

print(f"\n🏆 Best Configuration: {best_config['name']}")
print(f"   Test Accuracy: {best_config['test_accuracy']:.1%}")
print(f"   Test MAE: {best_config['test_mae']:.3f}")
print(f"   Optimal Threshold: {best_config['best_threshold']:.2f}")
print(f"   Home Cover Rate: {best_config['home_cover_rate']:.1%}")
print(f"   Overfitting (MAE diff): {best_config['mae_diff']:.3f}")

print(f"\n📊 All Configurations:")
print("-"*70)
for idx, row in results_df.iterrows():
    print(f"  {row['name']:<25} | Acc: {row['test_accuracy']:.1%} | MAE: {row['test_mae']:.3f} | Threshold: {row['best_threshold']:.2f}")

print(f"\n{'='*70}")

