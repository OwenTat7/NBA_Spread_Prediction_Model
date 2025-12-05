# DraftKings Spread Integration

## Overview
The model now incorporates DraftKings sportsbook spreads as features. DraftKings spreads reflect public/market opinion and can provide valuable signal for predicting game outcomes.

## Implementation Details

### 1. Data Extraction

#### Historical Games (`fetch_historical_data.py`)
- **Opening Spread**: Extracted from `pointSpread.home.open.line` in the ESPN API
- **Closing Spread**: Extracted from `pointSpread.home.close.line` in the ESPN API
- Both spreads are converted to home-team perspective
- Falls back to simple `spread` field if `pointSpread` structure not available

#### Upcoming Games (`predict_upcoming_games.py`)
- **Opening Spread**: Extracted from `pointSpread.home.open.line`
- **Current Spread**: Extracted from `pointSpread.home.close.line` (for upcoming games, "close" represents the current line)
- Both converted to home-team perspective

### 2. Feature Engineering (`run_full_pipeline.py`)

Three new DraftKings-based features are added:

#### `dk_spread`
- **Definition**: DraftKings closing spread (for historical games) or current spread (for upcoming games)
- **Purpose**: Direct market expectation of the game outcome
- **Calculation**: 
  - Historical: Uses `closing_spread` from API
  - Upcoming: Uses `current_spread` from API
  - Fallback: ELO-based estimate if no DraftKings data available

#### `dk_spread_diff`
- **Definition**: Difference between DraftKings spread and our ELO-based spread estimate
- **Formula**: `dk_spread - elo_spread_estimate`
- **Purpose**: Captures where the market disagrees with our ELO model
  - **Positive**: Market favors home team more than ELO suggests
  - **Negative**: Market favors away team more than ELO suggests
- **Value**: This feature can identify market inefficiencies or public sentiment shifts

#### `dk_spread_move`
- **Definition**: Line movement from opening to closing/current spread
- **Formula**: `closing_spread - opening_spread`
- **Purpose**: Captures market sentiment shifts
  - **Positive**: Line moved toward home team (money coming in on home team)
  - **Negative**: Line moved toward away team (money coming in on away team)
- **Value**: Sharp money movements often indicate informed betting

### 3. Target Variable

The model still uses `residual = margin - closing_spread` as the target, where `closing_spread` is now:
1. DraftKings closing spread (if available)
2. DraftKings current spread (for upcoming games)
3. DraftKings opening spread (if no closing/current)
4. ELO-based estimate (fallback)

### 4. Model Training

- All three DraftKings features (`dk_spread`, `dk_spread_diff`, `dk_spread_move`) are included as model inputs
- These features are NOT excluded from training (unlike `closing_spread` which is used for target calculation)
- The model learns how to combine ELO ratings, team stats, injuries, and market information (DraftKings spreads) to predict residuals

## Expected Benefits

1. **Market Information**: Incorporates public opinion and sharp money movements
2. **Market Efficiency**: DraftKings spreads are highly efficient; using them as features helps the model understand market expectations
3. **Disagreement Signal**: `dk_spread_diff` identifies games where our model disagrees with the market, potentially finding value
4. **Line Movement**: `dk_spread_move` captures late money and injury news impacts

## Usage

The DraftKings features are automatically:
- Extracted from ESPN API when fetching historical or upcoming games
- Added to feature set during model training
- Included in predictions for upcoming games

No additional configuration needed - the model will use DraftKings data when available and fall back to ELO estimates when not.

## Data Quality

- **Coverage**: DraftKings spreads are available for most games via ESPN API
- **Accuracy**: Spreads are from DraftKings (provider ID "100"), a major sportsbook
- **Timing**: 
  - Historical games: Closing spread represents final line before game
  - Upcoming games: Current spread represents latest available line

## Notes

- The model still calculates ELO-based spreads for comparison and fallback
- If DraftKings data is unavailable, the model gracefully falls back to ELO estimates
- All spreads are normalized to home-team perspective for consistency

