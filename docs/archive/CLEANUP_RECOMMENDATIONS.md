# Cleanup Recommendations - Prototype Files

## Files Safe to Remove

### Test/Prototype Scripts
These were created during development and are no longer needed:

1. **`test_espn_odds_api.py`**
   - Purpose: Test script to explore ESPN API odds data
   - Status: ✅ Functionality integrated into main scripts
   - Safe to delete: YES

2. **`fetch_historical_data_test.py`**
   - Purpose: Test version for fetching historical data
   - Status: ✅ Replaced by `01_data_collection_fetch_historical_games.py`
   - Safe to delete: YES

3. **`test_prediction_pipeline.py`**
   - Purpose: Quick test of prediction pipeline
   - Status: ✅ Not needed - use `run_full_pipeline.py` instead
   - Safe to delete: YES

4. **`run_prediction_pipeline.py`**
   - Purpose: Old version of prediction pipeline (references notebook)
   - Status: ✅ Replaced by `run_full_pipeline.py`
   - Safe to delete: YES

### Duplicate Background Scripts
These are duplicate versions with functionality now in main scripts:

5. **`add_injuries_background.py`**
   - Purpose: Background/non-interactive version of injury collection
   - Status: ✅ `02_data_collection_add_injuries.py` supports non-interactive mode
   - Safe to delete: YES

6. **`collect_injuries_background.py`**
   - Purpose: Another background version of injury collection
   - Status: ✅ Duplicate functionality
   - Safe to delete: YES

### Potentially Duplicate Utility
7. **`fetch_historical_spreads.py`**
   - Purpose: Test script to check if ESPN API provides spread data
   - Status: ⚠️ Functionality now in `01_data_collection_fetch_historical_games.py`
   - Safe to delete: PROBABLY (check if it has unique utility)

---

## Files to Keep

### Core Scripts
- `00_data_collection_run_all.py` - Master data collection
- `01_data_collection_fetch_historical_games.py` - Fetch games
- `02_data_collection_add_injuries.py` - Add injuries
- `run_full_pipeline.py` - Train model
- `predict_upcoming_games.py` - Make predictions
- `daily_predictions.py` - Daily automation

### Utility Scripts (Keep)
- `backfill_historical_spreads.py` - Utility to backfill missing spreads
- `check_spreads.py` - Verify spread data quality
- `analyze_injury_features.py` - Analysis tool for injury features
- `injury_features.py` - Core module (imported by other scripts)

---

## Summary

**Total files safe to remove: 6-7**
- 4 test/prototype scripts
- 2 duplicate background scripts
- 1 potentially duplicate utility

**Recommendation**: Remove the 6 confirmed duplicates/test files. Keep `fetch_historical_spreads.py` for now unless you confirm it's not needed.

