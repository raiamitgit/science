#!/usr/bin/env bash
# Builds and deploys af3_batch.py as a Cloud Run Job.
set -euo pipefail

# Load settings from a passed .env path or from .env next to this script.
SCRIPT_DIR="$(dirname "$0")"
if (($#)); then
  [[ "$1" == */* ]] || set -- "./$1"
  source "$1"
elif [[ -f "$SCRIPT_DIR/.env" ]]; then
  source "$SCRIPT_DIR/.env"
fi

# Fail before building if the project or service account is missing or empty.
: "${PROJECT:?Set PROJECT in .env or environment}"
: "${SERVICE_ACCOUNT:?Set SERVICE_ACCOUNT in .env or environment}"

gcloud run jobs deploy "${JOB:-af3-batch}" \
  --source "$SCRIPT_DIR" \
  --project "$PROJECT" \
  --region "${REGION:-us-central1}" \
  --tasks 1 \
  --cpu 1 \
  --memory 2Gi \
  --task-timeout "${TASK_TIMEOUT:-24h}" \
  --max-retries 0 \
  --service-account "$SERVICE_ACCOUNT"
