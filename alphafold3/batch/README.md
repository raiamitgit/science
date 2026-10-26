# AlphaFold 3 Batch Prediction

[AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w) is available on Google Cloud Gemini Platform for commercial use. Each online endpoint replica folds one biomolecular complex at a time. This batch runner sends a folder of AlphaFold 3 prediction requests to a deployed endpoint from a local terminal, a Python script, or Cloud Run Jobs:

* Reads a folder of JSON prediction requests from Google Cloud Storage.
* Validates every request and verifies empty output folders with an upfront dry run before starting any folds.
* Reads the endpoint's GPU replica count automatically, runs folds in parallel, and retries when replicas are busy or transient storage errors occur.
* Polls Google Cloud Storage if a long fold outlasts the client HTTP connection, recording predictions that finish on the endpoint as succeeded.

## Prerequisites

Check that four prerequisites are in place before running a batch:

1. Deploy an AlphaFold 3 Dedicated Endpoint on Gemini Platform with its inference timeout set to 3,600 seconds, since standard shared endpoints time out after 10 minutes. Private Service Connect endpoints are also supported when private DNS and TLS certificate trust are configured for the caller environment.
2. Grant `roles/aiplatform.user` on the endpoint project and `roles/storage.objectViewer` on the input and output Cloud Storage buckets to the batch runner identity, whether that is your user account or the Cloud Run job service account. Uploading input files to the bucket requires write access such as `roles/storage.objectUser`.
3. Grant `roles/storage.objectUser` on the output bucket to the endpoint's serving service account, which defaults to `custom-online-prediction@<TENANT_PROJECT>-tp.iam.gserviceaccount.com` unless a custom service account was attached during deployment, so it can check output folder emptiness and write results. Grant `roles/storage.objectViewer` on any separate bucket holding precomputed alignments or templates. You can find this service account in your endpoint logs or in the permission error returned by a live prediction before bucket access is granted.
4. Sign in to the Google Cloud CLI for local terminal runs, which keeps local execution on the Python standard library without extra package installs. Cloud Run Jobs authenticate directly through the metadata server using the attached service account.

## Preparing the Dataset

Store your batch input in Google Cloud Storage as a folder of JSON files, with one prediction request per file. Each file holds one [AlphaFold 3 input specification](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md) and its own empty output folder in Google Cloud Storage so results do not collide or overwrite earlier predictions. Replace the placeholder bucket name in the sample files and upload them to Google Cloud Storage:

```bash
for f in sample_requests/*.json; do
  sed "s/my-gcs-bucket/YOUR_ACTUAL_BUCKET/g" "$f" \
    | gcloud storage cp - "gs://YOUR_ACTUAL_BUCKET/af3_inputs/$(basename "$f")"
done
```

## Running the Batch

Copy the full prediction URL from your endpoint's Sample Request page in the Google Cloud console, noting that Dedicated Endpoints use their dedicated `*.prediction.vertexai.goog` host rather than `aiplatform.googleapis.com`. Run a dry run first to confirm that every request passes validation and points to an empty output folder:

```bash
ENDPOINT_URL="https://HOST/v1/projects/PROJECT_ID/locations/LOCATION/endpoints/ENDPOINT_ID:predict"

python3 af3_batch.py --input gs://my-gcs-bucket/af3_inputs --endpoint-url "$ENDPOINT_URL" --dry-run
```

Once the dry run passes, run the batch, or add the resume flag after an interruption to skip requests whose output folders already hold finished predictions:

```bash
python3 af3_batch.py --input gs://my-gcs-bucket/af3_inputs --endpoint-url "$ENDPOINT_URL"

# Resume an interrupted batch, skipping completed output folders:
python3 af3_batch.py --input gs://my-gcs-bucket/af3_inputs --endpoint-url "$ENDPOINT_URL" --resume
```

From Python, import the module and call the runner directly:

```python
import af3_batch

af3_batch.run_batch("gs://my-gcs-bucket/af3_inputs", ENDPOINT_URL, dry_run_only=True)
results = af3_batch.run_batch("gs://my-gcs-bucket/af3_inputs", ENDPOINT_URL)
results = af3_batch.run_batch("gs://my-gcs-bucket/af3_inputs", ENDPOINT_URL, resume=True)
```

For each completed request, the endpoint writes the prediction outputs directly to that request's output folder in Google Cloud Storage.

## Running on Cloud Run Jobs

Multi-hour batches can run as a serverless Cloud Run Job under a service account so the batch does not depend on an open terminal or an interactive login session. Using Cloud Build and Artifact Registry, an administrator configures the environment file and deploys the job once, while team members who only submit runs need `roles/run.jobsExecutorWithOverrides` on the job:

```bash
cp .env.example .env
# Edit .env with your PROJECT, SERVICE_ACCOUNT, and REGION, then deploy:
./deploy_cloud_run.sh
```

Start a batch execution in the background, or add the resume flag when restarting an interrupted run:

```bash
gcloud run jobs execute af3-batch \
  --project PROJECT_ID \
  --region REGION \
  --args="--input=gs://my-gcs-bucket/af3_inputs,--endpoint-url=$ENDPOINT_URL"

# Resume an interrupted batch:
gcloud run jobs execute af3-batch \
  --project PROJECT_ID \
  --region REGION \
  --args="--input=gs://my-gcs-bucket/af3_inputs,--endpoint-url=$ENDPOINT_URL,--resume"
```

Cloud Run prints the execution name right away. Follow live per-fold progress in Cloud Run Job logs, set up a Cloud Monitoring email alert on `run.googleapis.com/job/completed_execution_count`, or check overall execution status directly:

```bash
gcloud run jobs executions describe EXECUTION_NAME --project PROJECT_ID --region REGION
```
