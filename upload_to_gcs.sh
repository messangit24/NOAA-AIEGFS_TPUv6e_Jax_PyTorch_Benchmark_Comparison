#!/usr/bin/env bash
# ==============================================================================
# NOAA AIGEFS Benchmark Output GCS Uploader
# Uploads the 16-day forecast output and telemetry reports to gs://mb-noaa-eu/outputs/
# ==============================================================================

set -euo pipefail

DEST_BUCKET="gs://mb-noaa-eu"
RUN_ID="20261008_225646"
TARGET_DIR="${DEST_BUCKET}/outputs/${RUN_ID}_aiegfs"

FORECAST_FILE="/tmp/aigefs_forecasts/aigec00.t12z.forecast_16day_${RUN_ID}.nc"
REPORT_FILE="/home/admin_messan_altostrat_com/aiegfs/reports/aigefs_benchmark_report_${RUN_ID}.txt"
METRICS_FILE="/home/admin_messan_altostrat_com/aiegfs/reports/aigefs_benchmark_metrics_${RUN_ID}.json"

echo "======================================================================"
echo "   UPLOADING AIGEFS TPU v6e BENCHMARK OUTPUTS TO GCS"
echo "======================================================================"
echo "Target Destination: ${TARGET_DIR}/"
echo ""

# Verify local files exist
if [[ ! -f "${FORECAST_FILE}" ]]; then
    echo "ERROR: Forecast file not found: ${FORECAST_FILE}" >&2
    exit 1
fi

echo "1. Uploading 16-day forecast NetCDF dataset ($(du -h "${FORECAST_FILE}" | cut -f1))..."
gcloud storage cp "${FORECAST_FILE}" "${TARGET_DIR}/"

if [[ -f "${REPORT_FILE}" ]]; then
    echo "2. Uploading benchmark report txt..."
    gcloud storage cp "${REPORT_FILE}" "${TARGET_DIR}/"
fi

if [[ -f "${METRICS_FILE}" ]]; then
    echo "3. Uploading benchmark metrics json..."
    gcloud storage cp "${METRICS_FILE}" "${TARGET_DIR}/"
fi

echo ""
echo "======================================================================"
echo "   VERIFYING UPLOADED GCS ARTIFACTS"
echo "======================================================================"
gcloud storage ls -l "${TARGET_DIR}/"
echo ""
echo "Upload complete and ready for data science verification!"
