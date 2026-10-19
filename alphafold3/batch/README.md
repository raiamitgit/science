# AlphaFold 3 Batch Prediction

[AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w) is available on Google Cloud Gemini Enterprise Agent Platform for commercial use. It supports interactive online predictions that fold one biomolecular complex at a time. This script is a thin wrapper that runs a batch of AlphaFold 3 predictions against a deployed online endpoint. You can run it from a local terminal, a Python script, or a batch job runner such as Cloud Run Jobs:

* Reads a batch of prediction requests from a Google Cloud Storage directory.
* Runs a dry run across every request before starting any predictions, checking validity of each request.
* Detects the endpoint's GPU replica count automatically, sends requests in parallel, and retries when replicas are busy.
* Polls the output folder in Google Cloud Storage if a long fold exceeds the client HTTP timeout, so jobs that finish on the endpoint are still recorded as succeeded.

## Prerequisites

Before running a batch job, make sure three prerequisites are in place:

1. A running AlphaFold 3 online endpoint deployed on Gemini Enterprise Agent Platform.
2. An existing Google Cloud Storage bucket where the caller and the endpoint service account has read/write access.
3. Google Cloud credentials configured through the gcloud CLI; alternatively the script can be reconfigured to use service account credentials too.

## Preparing the Dataset

Organize your batch input in Google Cloud Storage as a folder of JSON files, with one prediction request per file. Each request follows the Google Cloud AlphaFold 3 prediction schema, containing one [AlphaFold 3 input specification](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md) and a dedicated output folder in Google Cloud Storage. Every file must point to its own empty output folder so results do not collide or overwrite existing predictions. See the sample files in the repo. Replace the placeholder bucket name with your bucket and upload the folder to Google Cloud Storage before running:

```bash
for f in sample_requests/*.json; do
  sed "s/my-gcs-bucket/YOUR_ACTUAL_BUCKET/g" "$f" \
    | gcloud storage cp - "gs://YOUR_ACTUAL_BUCKET/af3_inputs/$(basename "$f")"
done
```

## Running the Batch

Copy the prediction URL from your endpoint's Sample Request page in the Google Cloud console. Run a dry run first to confirm that all input payloads pass schema validation and that every target output folder in Google Cloud Storage is empty:

```bash
python3 af3_batch.py \
  --input gs://my-gcs-bucket/af3_inputs \
  --endpoint-url "https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT/locations/us-central1/endpoints/YOUR_ENDPOINT:predict" \
  --dry-run
```

Once the dry run passes, run the batch job:

```bash
python3 af3_batch.py \
  --input gs://my-gcs-bucket/af3_inputs \
  --endpoint-url "https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT/locations/us-central1/endpoints/YOUR_ENDPOINT:predict"
```

From Python, import the module and call the batch runner directly:

```python
import af3_batch

af3_batch.run_batch("gs://my-gcs-bucket/af3_inputs", ENDPOINT_URL, dry_run_only=True)
results = af3_batch.run_batch("gs://my-gcs-bucket/af3_inputs", ENDPOINT_URL)
```

For each completed request, the endpoint writes the prediction outputs directly to that request's output folder in Google Cloud Storage.
