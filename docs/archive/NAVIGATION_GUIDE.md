# Navigation Guide

Quick reference for finding files and information in this project.

## 🗂️ Directory Structure

### `data/` - Data Files
| File | Description | Usage |
|------|-------------|-------|
| `nba_games.csv` | Main dataset (3,160 games) | ⭐ **Primary data file** |
| `nba_games_with_injuries.csv` | Games with injury features | For injury analysis only |
| `nba_games_test.csv` | Test dataset (213 games) | Development/testing |
| `*_temp.csv`, `*_sample.csv` | Temporary/sample files | Not for production use |

### `docs/` - Documentation

#### ⭐ Start Here
| File | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `QUICK_START.md` | Getting started guide |
| `INJURY_FEATURE_ANALYSIS_SUMMARY.md` | ⭐ **Key findings on injuries** |

#### Analysis Reports
| File | Content |
|------|---------|
| `ACCURACY_REPORT.md` | Model performance metrics (65% accuracy) |
| `PERFORMANCE_COMPARISON.md` | With vs without injuries comparison |
| `INJURY_MODEL_RESULTS.md` | Results with injury features |
| `CORRELATION_TEST_RESULTS.md` | Multicollinearity analysis |

#### Guides & Status
| File | Content |
|------|---------|
| `INJURY_INTEGRATION_GUIDE.md` | How injuries were integrated |
| `INJURY_FEATURES_SUMMARY.md` | Injury feature descriptions |
| `MODEL_READINESS_TIMELINE.md` | Development timeline |
| `STATUS.md`, `PROGRESS.md` | Project status updates |
| `FINAL_STATUS.md` | Current status summary |

### `scripts/` - Python Scripts

#### Main Scripts
| Script | Purpose | Usage |
|--------|---------|-------|
| `run_full_pipeline.py` | ⭐ **Main pipeline** | `python scripts/run_full_pipeline.py` |
| `analyze_injury_features.py` | Injury feature analysis | Compare configurations |
| `fetch_historical_data.py` | Data collection | Fetch from ESPN API |

#### Utility Scripts
| Script | Purpose |
|--------|---------|
| `test_prediction_pipeline.py` | Pipeline testing |
| `add_injuries_to_pipeline.py` | Add injury features |
| `add_injuries_background.py` | Background injury collection |
| `collect_injuries_background.py` | Batch injury collection |
| `injury_features.py` | Injury feature module |
| `fetch_historical_data_test.py` | Test data collection |

### `notebooks/` - Jupyter Notebooks
| Notebook | Purpose |
|----------|---------|
| `nba_spread_prediction.ipynb` | ⭐ Main prediction notebook |
| `nba_data_collection.ipynb` | Data collection notebook |

### `logs/` - Log Files
| File | Content |
|------|---------|
| `fetch_log.txt` | Data collection logs |
| `injury_collection.log` | Injury data collection logs |
| `TIMELINE_SUMMARY.txt` | Timeline summary |

### `models/` - Saved Models
*Generated when pipeline runs* - Contains trained models and feature importance.

---

## 🎯 Common Tasks

### I want to...

#### ...run the model
```bash
python scripts/run_full_pipeline.py
```
See: `docs/QUICK_START.md`

#### ...understand the results
- Start: `docs/INJURY_FEATURE_ANALYSIS_SUMMARY.md`
- Details: `docs/ACCURACY_REPORT.md`
- Comparison: `docs/PERFORMANCE_COMPARISON.md`

#### ...add new data
```bash
python scripts/fetch_historical_data.py
```
See: `scripts/fetch_historical_data.py`

#### ...analyze injury features
```bash
python scripts/analyze_injury_features.py
```
See: `docs/INJURY_FEATURE_ANALYSIS_SUMMARY.md`

#### ...understand the code
- Pipeline: `scripts/run_full_pipeline.py`
- Notebook: `notebooks/nba_spread_prediction.ipynb`
- Features: `scripts/injury_features.py`

#### ...check project status
- Current status: `docs/FINAL_STATUS.md`
- Timeline: `docs/MODEL_READINESS_TIMELINE.md`
- Progress: `docs/PROGRESS.md`

---

## 📊 Key Results Summary

### Model Performance

**Best Configuration:**
- Cover Accuracy: **56.1%** (cross-validated)
- MAE: **11.33** points
- Features: **29** (no injuries)

**With Injuries:**
- Cover Accuracy: 55.9%
- MAE: 11.33 points
- Status: ❌ Not recommended

### Documentation Flow

```
Start Here
    ↓
docs/INJURY_FEATURE_ANALYSIS_SUMMARY.md  (Key findings)
    ↓
docs/ACCURACY_REPORT.md  (Performance details)
    ↓
docs/PERFORMANCE_COMPARISON.md  (With vs without)
    ↓
docs/INJURY_INTEGRATION_GUIDE.md  (Technical details)
```

---

## 🔍 Finding Specific Information

### Where to find...

**Model accuracy numbers**
→ `docs/ACCURACY_REPORT.md`

**Why injuries don't help**
→ `docs/INJURY_FEATURE_ANALYSIS_SUMMARY.md`

**How to run the model**
→ `docs/QUICK_START.md` or `README.md`

**Feature engineering details**
→ `notebooks/nba_spread_prediction.ipynb` or `scripts/run_full_pipeline.py`

**Data collection process**
→ `scripts/fetch_historical_data.py` or `notebooks/nba_data_collection.ipynb`

**Injury feature implementation**
→ `docs/INJURY_INTEGRATION_GUIDE.md` or `scripts/injury_features.py`

**Correlation analysis**
→ `docs/CORRELATION_TEST_RESULTS.md` or `docs/MULTICOLLINEARITY_ANALYSIS.md`

---

## 🎓 Learning Path

1. **Read** `README.md` - Overview
2. **Read** `docs/INJURY_FEATURE_ANALYSIS_SUMMARY.md` - Key findings
3. **Run** `scripts/run_full_pipeline.py` - See it in action
4. **Explore** `notebooks/nba_spread_prediction.ipynb` - Detailed analysis
5. **Review** `docs/ACCURACY_REPORT.md` - Performance metrics

---

**Last Updated:** December 1, 2025

