# Multicollinearity Analysis: Injuries vs Spread Movement

## The Question
**Will adding injury features cause multicollinearity since injuries are already reflected in spread movement?**

## The Relationship Chain

```
Injury News → Bookmakers Adjust Spreads → Spread Movement → Our Model
     ↓                                              ↑
     └────────── Direct Injury Features ───────────┘
```

## When Multicollinearity Occurs

Multicollinearity happens when:
1. **High correlation** between features (r > 0.8-0.9)
2. **Redundant information** - one feature can predict another
3. **Unstable coefficients** - model can't determine which feature matters

## The Reality Check

### ✅ **Injuries ARE often in spreads, BUT:**

1. **Timing Differences**
   - Injuries announced → Spread moves (hours/days later)
   - Our model uses **pre-game state** - injuries might be known before final spread
   - **Edge case**: Early injury news before line moves

2. **Information Asymmetry**
   - Bookmakers might **over-react** or **under-react** to injuries
   - Model can learn: "When star out but spread only moved 2 points, actual impact is 4 points"
   - **Value**: Model finds mispricings

3. **Non-Linear Interactions**
   - Injury + rest days + home court = different impact than spread alone
   - Example: "Star out + back-to-back + away game" might be worse than spread suggests
   - **Value**: Interactions spread movement doesn't capture

4. **Incomplete Information**
   - Spreads reflect **market consensus**, not perfect information
   - Some injuries have **uncertain impact** (day-to-day, minutes restrictions)
   - **Value**: Model can learn actual impact vs market expectation

5. **Historical Patterns**
   - Model learns: "When Player X is out, team performs Y points worse than spread suggests"
   - Spreads are **forward-looking**, injuries are **factual**
   - **Value**: Historical injury impact patterns

## Testing Strategy

### Option 1: Correlation Analysis
```python
# Check correlation between injury features and spread_move
correlation = df[['players_out_home', 'spread_move', 'closing_spread']].corr()
# If r > 0.8, consider removing or combining
```

### Option 2: Feature Importance
```python
# Train model with and without injury features
# If injuries don't improve performance, they're redundant
# If they do improve, they add value beyond spreads
```

### Option 3: VIF (Variance Inflation Factor)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF > 10 indicates multicollinearity
# VIF 5-10 is moderate concern
```

### Option 4: Residual Analysis
```python
# Train model on spread_move alone
# Then add injury features
# If residuals don't improve, injuries are redundant
```

## Recommended Approach

### **Build with Multicollinearity Checks Built-In**

1. **Add injury features**
2. **Calculate correlations** with spread features
3. **Test feature importance** (with/without injuries)
4. **Use regularization** (Ridge/Lasso already handles this)
5. **Tree models** (LightGBM) are less sensitive to multicollinearity

### **Smart Feature Engineering**

Instead of raw injury counts, create **interaction features**:

```python
# Instead of: players_out_home (might correlate with spread_move)
# Create: injury_impact_beyond_spread = (players_out_home * elo_home) - spread_move
# This captures: "Injury impact not reflected in spread"
```

### **Feature Selection**

Use **feature importance** to drop redundant features:
- If `spread_move` has importance 0.8 and `players_out_home` has 0.05
- The injury feature adds little beyond spread movement
- Can drop it or combine it

## Expected Outcome

### **Best Case Scenario:**
- Injuries add **5-10% accuracy improvement**
- Low correlation with spreads (r < 0.5)
- Model learns: "Injuries matter even after accounting for spread movement"

### **Worst Case Scenario:**
- High correlation (r > 0.8)
- No accuracy improvement
- **Solution**: Drop injury features or combine them

### **Most Likely:**
- Moderate correlation (r = 0.4-0.7)
- **Small but meaningful** improvement (2-5% accuracy)
- Injuries capture **nuances** spreads miss (timing, interactions)

## My Recommendation

**✅ YES, add injuries, but with safeguards:**

1. **Calculate correlations** during feature engineering
2. **Test with/without** injury features
3. **Use regularization** (already in your pipeline)
4. **Monitor feature importance** - drop if redundant
5. **Create interaction features** that capture injury impact beyond spreads

**Why this works:**
- LightGBM handles multicollinearity well (tree-based)
- Regularization (Ridge) already in your pipeline
- Feature importance will show if injuries add value
- Easy to remove if they don't help

## Implementation Plan

I'll build it with:
1. ✅ Injury feature extraction
2. ✅ Correlation analysis (report correlations)
3. ✅ Feature importance comparison
4. ✅ Option to exclude if redundant
5. ✅ Interaction features (injury × other features)

**Result**: You'll know if injuries help, and if they cause multicollinearity, we'll catch it and handle it.

---

**Bottom Line**: The concern is valid, but we can test it. Injuries might add value through timing, interactions, and information asymmetry. If they don't, we'll know and can remove them.


