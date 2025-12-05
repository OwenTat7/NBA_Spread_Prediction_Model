# Final Training Dataset - Complete Summary

## Overview
After running the full data collection pipeline, here's exactly what data the model trains on.

---

## 📊 SOURCE DATA FILE

**File**: `data/nba_games_with_injuries.csv`

**Raw Columns** (from data collection):
- `game_id` - ESPN game identifier
- `date` - Game date
- `name` - Game description
- `status` - Game status
- `completed` - Boolean completion flag
- `venue` - Venue name
- `home_team_id`, `home_team_name`, `home_team_abbrev`
- `away_team_id`, `away_team_name`, `away_team_abbrev`
- `home_score`, `away_score`
- `home_winner`, `away_winner`
- `point_differential` - Calculated (home_score - away_score)
- `opening_spread` - Opening betting line (from ESPN API)
- `over_under` - Total points line
- `favorite_id` - Favorite team ID
- **Injury Features** (13 columns):
  - `players_out_home`, `players_out_away` - Count of players out
  - `players_dtd_home`, `players_dtd_away` - Count day-to-day
  - `injury_severity_home`, `injury_severity_away` - Severity scores
  - `star_out_home`, `star_out_away` - Binary (any star out)
  - `total_injury_impact_home`, `total_injury_impact_away` - Impact scores
  - `injury_severity_diff` - Home - Away severity
  - `players_out_diff` - Home - Away count
  - `total_injury_impact_diff` - Home - Away impact

---

## 🔧 FEATURE ENGINEERING

The `build_features()` function adds the following engineered features:

### 1. **Rest Days** (3 features)
- `rest_days_home` - Days since home team's last game
- `rest_days_away` - Days since away team's last game
- `rest_diff` - Home rest days - Away rest days

### 2. **Rolling Statistics** (18 features)
For windows of 3, 5, and 10 games:
- `home_margin_avg_3`, `home_margin_avg_5`, `home_margin_avg_10` - Home team's average margin
- `away_margin_avg_3`, `away_margin_avg_5`, `away_margin_avg_10` - Away team's average margin
- `home_points_for_avg_3`, `home_points_for_avg_5`, `home_points_for_avg_10` - Home team's average points scored
- `away_points_for_avg_3`, `away_points_for_avg_5`, `away_points_for_avg_10` - Away team's average points scored
- `home_points_against_avg_3`, `home_points_against_avg_5`, `home_points_against_avg_10` - Home team's average points allowed
- `away_points_against_avg_3`, `away_points_against_avg_5`, `away_points_against_avg_10` - Away team's average points allowed

### 3. **ELO Ratings** (3 features)
- `elo_home` - Home team's ELO rating (before game)
- `elo_away` - Away team's ELO rating (before game)
- `elo_diff` - Home ELO - Away ELO

### 4. **Temporal Features** (4 features)
- `home_court` - Always 1 (home advantage indicator)
- `day_of_week` - 0-6 (Monday-Sunday)
- `month` - 1-12
- `is_weekend` - Binary (1 if Saturday/Sunday)

### 5. **Interaction Features** (1 feature)
- `elo_diff_x_rest_diff` - ELO difference × Rest difference

### 6. **Spread Features** (1 feature)
- `closing_spread` - Final spread used for training
  - **Priority**: `current_spread` > `opening_spread` > Elo-based estimate `(elo_diff / 25)`
  - Converted to home-team perspective

### 7. **Target Variables** (calculated, not used as features)
- `margin` - Actual game margin (home_score - away_score)
- `residual` - **TARGET**: margin - closing_spread (what model predicts)
- `cover` - Binary: 1 if home covered, 0 if away covered

---

## 🎯 FINAL MODEL INPUTS

### Features Used for Training (42 total)

**Excluded from training** (metadata/targets):
- `game_id`, `date`, `home_team_id`, `away_team_id`
- `home_team_name`, `away_team_name`
- `home_score`, `away_score`, `margin`
- `residual` (target), `cover`, `closing_spread`
- `opening_spread`, `status`, `completed`, `venue`, etc.

**Included in training** (all numeric features):

#### **ELO Features** (3)
1. `elo_home`
2. `elo_away`
3. `elo_diff`

#### **Rest Days** (3)
4. `rest_days_home`
5. `rest_days_away`
6. `rest_diff`

#### **Rolling Margins** (6)
7. `home_margin_avg_3`
8. `home_margin_avg_5`
9. `home_margin_avg_10`
10. `away_margin_avg_3`
11. `away_margin_avg_5`
12. `away_margin_avg_10`

#### **Rolling Points For** (6)
13. `home_points_for_avg_3`
14. `home_points_for_avg_5`
15. `home_points_for_avg_10`
16. `away_points_for_avg_3`
17. `away_points_for_avg_5`
18. `away_points_for_avg_10`

#### **Rolling Points Against** (6)
19. `home_points_against_avg_3`
20. `home_points_against_avg_5`
21. `home_points_against_avg_10`
22. `away_points_against_avg_3`
23. `away_points_against_avg_5`
24. `away_points_against_avg_10`

#### **Temporal Features** (4)
25. `home_court` (always 1)
26. `day_of_week`
27. `month`
28. `is_weekend`

#### **Interaction Features** (1)
29. `elo_diff_x_rest_diff`

#### **Injury Features** (13)
30. `players_out_home`
31. `players_out_away`
32. `players_dtd_home`
33. `players_dtd_away`
34. `injury_severity_home`
35. `injury_severity_away`
36. `star_out_home`
37. `star_out_away`
38. `total_injury_impact_home`
39. `total_injury_impact_away`
40. `injury_severity_diff`
41. `players_out_diff`
42. `total_injury_impact_diff`

---

## 🎯 TARGET VARIABLE

**Target**: `residual` = `margin` - `closing_spread`

- **What it represents**: How much the actual margin differs from the spread
- **Positive residual**: Home team beat the spread (home covers)
- **Negative residual**: Away team beat the spread (away covers)
- **Zero residual**: Push (exactly matched the spread)

**Model Goal**: Predict the residual to determine if home or away will cover the spread.

---

## 📈 DATA STATISTICS

After full data collection, you should have:
- **~3,000-4,000 games** (depending on date range)
- **42 features** used for training
- **All games with scores** (completed games only)
- **Spreads**: Mix of actual ESPN spreads and Elo-based estimates
- **Injuries**: Available for all games (fetched from API)

---

## 🔄 DATA FLOW

```
Raw Data Collection
  ├── fetch_historical_data.py
  │   └── nba_games.csv (games + spreads)
  │
  └── add_injuries_to_pipeline.py
      └── nba_games_with_injuries.csv (games + spreads + injuries)

Feature Engineering
  └── build_features() in run_full_pipeline.py
      ├── Calculates rest days
      ├── Calculates rolling stats (3, 5, 10 game windows)
      ├── Calculates ELO ratings
      ├── Adds temporal features
      ├── Creates interaction features
      └── Uses actual spreads (or Elo estimate if missing)

Model Training
  └── prepare_model_data()
      ├── Excludes metadata/target columns
      ├── Keeps all numeric features (42 total)
      └── Target: residual (margin - closing_spread)
```

---

## ✅ SUMMARY

**Final Training Dataset**:
- **Rows**: ~3,000-4,000 completed games
- **Features**: 42 numeric features
- **Target**: Residual (how much actual margin differs from spread)
- **Data Sources**:
  - Game scores: ESPN Scoreboard API
  - Spreads: ESPN API (preferred) or Elo-based estimate
  - Injuries: ESPN Summary API
  - Engineered: Rest days, rolling stats, ELO, temporal features

**Model Output**: Predicted residual
- Used to determine: Home covers vs Away covers
- Combined with spread to get: Predicted margin

