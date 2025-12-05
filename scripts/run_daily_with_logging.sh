#!/bin/bash
# Wrapper script for daily predictions with date-stamped logging

SCRIPT_DIR="/Users/owentatlonghari/Library/CloudStorage/OneDrive-UniversityofDelaware-o365/NBA_Prediction_Model"
LOG_DIR="${SCRIPT_DIR}/logs"
DATE=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/daily_predictions_${DATE}.log"
ERROR_LOG="${LOG_DIR}/daily_predictions_error_${DATE}.log"

cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Run the prediction script and log with timestamp
{
    echo "=========================================="
    echo "Daily NBA Predictions - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    /opt/anaconda3/bin/python3 "${SCRIPT_DIR}/scripts/daily_predictions.py"
    echo ""
    echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$LOG_FILE" 2>> "$ERROR_LOG"

exit 0

