#!/usr/bin/env python3
"""
Analyze which injury features are most useful
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import sys
import warnings
warnings.filterwarnings('ignore')

# Import pipeline functions
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from run_full_pipeline import build_features, prepare_model_data, load_game_data

print("="*70)
print("INJURY FEATURE ANALYSIS")
print("="*70)

# Load data
games_df = load_game_data()

# Build features
features_df_all = build_features(games_df, include_spread=True)

# Get injury features
all_injury_features = [c for c in features_df_all.columns if 'injury' in c.lower() or 
                       'players_out' in c.lower() or 'players_dtd' in c.lower() or 
                       'star_out' in c.lower()]

print(f"\nFound {len(all_injury_features)} injury features")

# Test configurations
configs = [
    ("No Injuries", []),
    ("All Injuries", all_injury_features),
    ("Diff Features Only", ['injury_severity_diff', 'players_out_diff', 'total_injury_impact_diff']),
    ("Severity Only", ['injury_severity_home', 'injury_severity_away', 'injury_severity_diff']),
    ("Star Players Only", ['star_out_home', 'star_out_away']),
    ("Top Correlation", ['star_out_home', 'injury_severity_away']),
]

results = []
tscv = TimeSeriesSplit(n_splits=3)

for config_name, injury_subset in configs:
    # Create feature set
    if injury_subset:
        injury_subset = [f for f in injury_subset if f in features_df_all.columns]
        features_to_remove = [f for f in all_injury_features if f not in injury_subset]
        features_df = features_df_all.drop(columns=features_to_remove)
    else:
        features_to_remove = all_injury_features
        features_df = features_df_all.drop(columns=features_to_remove)
    
    # Prepare data
    X, y, feature_names, metadata = prepare_model_data(features_df, target='residual')
    
    # Cross-validation
    cv_maes = []
    cv_cover_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        meta_val = metadata.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names)
        
        params = {
            "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
            "learning_rate": 0.05, "num_leaves": 31, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20,
            "verbose": -1, "seed": 42
        }
        
        model = lgb.train(params, dtrain, num_boost_round=500,
                         valid_sets=[dval], callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        
        pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        mae = mean_absolute_error(y_val, pred_val)
        cv_maes.append(mae)
        
        # Cover accuracy
        pred_cover = (pred_val > 0).astype(int)
        true_cover = (meta_val['residual'].values > 0).astype(int)
        cover_acc = (pred_cover == true_cover).mean()
        cv_cover_accs.append(cover_acc)
    
    avg_mae = np.mean(cv_maes)
    std_mae = np.std(cv_maes)
    avg_cover = np.mean(cv_cover_accs)
    std_cover = np.std(cv_cover_accs)
    
    results.append({
        'config': config_name,
        'n_features': len(injury_subset),
        'cv_mae': avg_mae,
        'cv_mae_std': std_mae,
        'cv_cover': avg_cover,
        'cv_cover_std': std_cover,
        'features': injury_subset
    })

# Sort by CV MAE
results.sort(key=lambda x: x['cv_mae'])

print("\n" + "="*70)
print("RESULTS (Cross-Validated)")
print("="*70)
print(f"{'Configuration':<25} {'# Inj':<6} {'CV MAE':<12} {'CV Cover Acc':<15}")
print("-"*70)

for r in results:
    marker = "🏆" if r == results[0] else "  "
    print(f"{marker} {r['config']:<23} {r['n_features']:<6} "
          f"{r['cv_mae']:>7.3f}±{r['cv_mae_std']:<4.3f}  "
          f"{r['cv_cover']:>6.1%}±{r['cv_cover_std']:<4.1%}")

print("\n" + "="*70)
print("BEST CONFIGURATION:")
print("="*70)
best = results[0]
print(f"Configuration: {best['config']}")
print(f"Features ({best['n_features']}): {best['features']}")
print(f"CV MAE: {best['cv_mae']:.3f} ± {best['cv_mae_std']:.3f}")
print(f"CV Cover Accuracy: {best['cv_cover']:.1%} ± {best['cv_cover_std']:.1%}")

# Feature importance analysis for best config
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Best Configuration)")
print("="*70)

if best['features']:
    # Train final model for importance
    features_df_best = features_df_all.drop(columns=[f for f in all_injury_features if f not in best['features']])
    X, y, feature_names, metadata = prepare_model_data(features_df_best, target='residual')
    
    avg_best_iter = int(np.mean([300, 300, 300]))  # Approximate
    final_model = lgb.train(
        {
            "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
            "learning_rate": 0.05, "num_leaves": 31, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20,
            "verbose": -1, "seed": 42
        },
        lgb.Dataset(X, label=y, feature_name=feature_names),
        num_boost_round=avg_best_iter
    )
    
    feature_importance = final_model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features:")
    print("-" * 60)
    injury_feats_in_best = best['features']
    for i, row in importance_df.head(20).iterrows():
        marker = "🏥" if row['feature'] in injury_feats_in_best else "  "
        print(f"{marker} {row['feature']:<40} {row['importance']:>10.0f}")
    
    injury_importance = importance_df[importance_df['feature'].isin(injury_feats_in_best)]['importance'].sum()
    total_importance = importance_df['importance'].sum()
    print(f"\n🏥 Selected injury features: {injury_importance/total_importance*100:.1f}% of total importance")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
if best['config'] == "No Injuries":
    print("✅ Use model WITHOUT injury features for best performance")
elif best['cv_mae'] < results[1]['cv_mae']:
    print(f"✅ Use model WITH selected injury features: {', '.join(best['features'])}")
    print(f"   Improvement: {results[1]['cv_mae'] - best['cv_mae']:.3f} MAE")
else:
    print("⚠️  Injury features provide minimal/no improvement")
print("="*70)

