#!/usr/bin/env python3
"""Runs a batch of AlphaFold 3 predictions on Google Cloud.

Validates every request JSON in a Cloud Storage folder with an upfront dry run, then dispatches
the folds in parallel across the endpoint's GPU replicas.
"""

import argparse
import concurrent.futures
import copy
import http.client
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# Base URL for the Cloud Storage JSON API used to list folders and read request files.
GCS_API = "https://storage.googleapis.com/storage/v1"
# The endpoint stops a fold after one hour, so no call, retry, or output wait needs to last longer.
FOLD_TIMEOUT_SEC = 3700

_token_lock = threading.Lock()
_token_cache = {"value": "", "expires": 0.0}


def access_token() -> str:
    """Returns a cached OAuth access token from the gcloud CLI."""
    with _token_lock:
        if time.monotonic() < _token_cache["expires"]:
            return _token_cache["value"]
        try:
            command = ["gcloud", "auth", "print-access-token"]
            token = subprocess.check_output(
                command, text=True, stderr=subprocess.DEVNULL, timeout=15
            ).strip()
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError(
                "No Google Cloud credentials found; run 'gcloud auth login'."
            ) from None
        # Tokens last an hour; refreshing every 5 minutes keeps long batches off expired tokens.
        _token_cache.update(value=token, expires=time.monotonic() + 300)
        return token


def call(url: str, body: dict | None = None, timeout: float = 60) -> tuple[int, str]:
    """Sends an authenticated GET, or a POST when body is set, and returns (status, text)."""
    data = None if body is None else json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {access_token()}", "Content-Type": "application/json"}
    try:
        request = urllib.request.Request(url, data, headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode(errors="replace")
    except urllib.error.HTTPError as error:
        # Return HTTP error bodies instead of raising so callers can report server error details.
        with error:
            return error.code, error.read().decode(errors="replace")
    except (OSError, http.client.HTTPException) as error:
        # Treat dropped sockets and client timeouts as status 0 so run_one can poll Cloud Storage.
        return 0, str(error)


def gcs_list(folder_uri: str, limit: int = 0) -> list[str]:
    """Returns gs:// URIs of the objects under a gs://<bucket>/<folder> prefix."""
    bucket, _, prefix = folder_uri.removeprefix("gs://").strip("/").partition("/")
    if not (folder_uri.startswith("gs://") and bucket and prefix):
        raise ValueError(f"Expected gs://<bucket>/<folder>, got {folder_uri!r}.")
    # Append a trailing slash so prefix matching stays inside folder_uri and skips sibling folders.
    prefix += "/"
    uris, page_token = [], ""
    while True:
        query = {"prefix": prefix}
        if limit:
            query["maxResults"] = limit
        if page_token:
            query["pageToken"] = page_token
        status, text = call(f"{GCS_API}/b/{bucket}/o?{urllib.parse.urlencode(query)}")
        if status != 200:
            raise RuntimeError(f"Listing {folder_uri} failed with HTTP {status}: {text[:300]}")
        page = json.loads(text)
        for item in page.get("items", []):
            uris.append(f"gs://{bucket}/{item['name']}")
        page_token = page.get("nextPageToken", "")
        if limit or not page_token:
            return uris


def load_requests(input_uri: str) -> list[dict]:
    """Loads all .json prediction requests from a gs://<bucket>/<folder>."""
    requests = []
    for uri in sorted(gcs_list(input_uri)):
        if not uri.endswith(".json"):
            continue
        bucket, _, name = uri.removeprefix("gs://").partition("/")
        quoted_name = urllib.parse.quote(name, safe="")
        status, text = call(f"{GCS_API}/b/{bucket}/o/{quoted_name}?alt=media")
        if status != 200:
            raise RuntimeError(f"Reading {uri} failed with HTTP {status}: {text[:300]}")
        requests.append(json.loads(text))
    if not requests:
        raise ValueError(f"No .json request files found in {input_uri}")
    return requests


def replica_count(endpoint_url: str) -> int:
    """Returns the endpoint's active replica count from the control plane, or 1 if unavailable."""
    path = urllib.parse.urlsplit(endpoint_url).path.removesuffix(":predict")
    parts = path.split("/")
    if len(parts) < 6 or parts[4] != "locations":
        return 1
    # The control plane sits on the regional aiplatform.googleapis.com host even for dedicated URLs.
    status, text = call(f"https://{parts[5]}-aiplatform.googleapis.com{path}")
    if status != 200:
        return 1
    endpoint = json.loads(text)
    traffic = endpoint.get("trafficSplit", {})
    replicas = 0
    # Only sum replicas on deployed models that receive traffic in trafficSplit.
    for model in endpoint.get("deployedModels", []):
        if traffic.get(model.get("id")):
            replicas += model.get("dedicatedResources", {}).get("maxReplicaCount", 0)
    return replicas or 1


def predict(endpoint_url: str, request: dict, timeout: float) -> tuple[int, str, float]:
    """Posts one request, retrying while every replica is busy (HTTP 429) for up to an hour."""
    give_up_at = time.monotonic() + FOLD_TIMEOUT_SEC
    while True:
        sent_at = time.monotonic()
        status, text = call(endpoint_url, request, timeout)
        if status != 429 or sent_at > give_up_at:
            return status, text, sent_at
        # Each replica runs one fold at a time; retry every second until an idle replica is picked.
        time.sleep(1)


def has_output(output_dir: str, deadline: float) -> bool:
    """Polls output_dir until the fold's top-level summary and structure files exist."""
    prefix = output_dir.rstrip("/") + "/"
    while True:
        try:
            top_files = []
            for uri in gcs_list(prefix):
                rel_path = uri.removeprefix(prefix).removesuffix(".zst")
                # Top-level outputs sit at output_dir/<job>/<file>; per-sample copies sit deeper.
                if rel_path.count("/") == 1:
                    top_files.append(rel_path)
            has_summary = any(name.endswith("_summary_confidences.json") for name in top_files)
            has_cif = any(name.endswith("_model.cif") for name in top_files)
            if has_summary and has_cif:
                return True
        except RuntimeError as error:
            print(f"Retrying output check for {prefix}: {error}")
        if time.monotonic() >= deadline:
            return False
        time.sleep(30)


def check_one(request: dict, endpoint_url: str) -> str | None:
    """Checks that one request's output_dir is empty in GCS and passes server-side dry_run."""
    # Copy the request before setting dry_run so the subsequent batch run executes the real fold.
    payload = copy.deepcopy(request)
    name = payload["instances"][0].get("name")
    # Support both top-level parameters (v3.0.4) and instances[0].parameters (multi-GPU builds).
    params = payload.get("parameters") or payload["instances"][0].get("parameters") or {}
    output_dir = str(params.get("output_dir") or "").rstrip("/")
    try:
        if gcs_list(output_dir, limit=1):
            return f"{name}: output_dir {output_dir} is not empty."
    except (ValueError, RuntimeError) as error:
        return f"{name}: {error}"
    params["dry_run"] = True
    status, text, _ = predict(endpoint_url, payload, timeout=120)
    if status != 200:
        return f"{name}: dry run failed with HTTP {status}: {text[:300]}"
    return None


def dry_run(requests: list[dict], endpoint_url: str, workers: int) -> list[str]:
    """Validates every request and its output_dir before any fold starts, returning any errors."""
    dirs, problems = [], []
    for number, request in enumerate(requests, 1):
        instances = request.get("instances") if isinstance(request, dict) else None
        if not (isinstance(instances, list) and len(instances) == 1
                and isinstance(instances[0], dict)):
            return [f"Request {number}: 'instances' must be a list with one AlphaFold 3 input."]
        params = request.get("parameters") or instances[0].get("parameters") or {}
        output_dir = str(params.get("output_dir") or "").rstrip("/")
        if output_dir:
            dirs.append(output_dir + "/")
    # With a trailing "/", any duplicate or nested folder sorts directly after its parent.
    dirs.sort()
    for outer, inner in zip(dirs, dirs[1:]):
        if inner.startswith(outer):
            outer_dir = outer.removesuffix("/")
            inner_dir = inner.removesuffix("/")
            problems.append(f"output_dir {inner_dir} overlaps output_dir {outer_dir}.")
    if problems:
        return problems

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(check_one, req, endpoint_url) for req in requests]
        for future in futures:
            problem = future.result()
            if problem:
                problems.append(problem)
    return problems


def run_one(request: dict, endpoint_url: str) -> dict:
    """Runs one request and returns its status, elapsed seconds, output folder, and error if any."""
    params = request.get("parameters") or request["instances"][0].get("parameters") or {}
    output_dir = params["output_dir"].rstrip("/")
    start = time.monotonic()
    error_message = ""
    # Any failure becomes a FAILED row, so one bad request cannot stop the rest of the batch.
    try:
        status, text, sent_at = predict(endpoint_url, request, FOLD_TIMEOUT_SEC)
        # The fold keeps running on the endpoint after the HTTP call drops, so wait for its output.
        deadline = sent_at + FOLD_TIMEOUT_SEC
        recovered = status in (0, 502, 503, 504) and has_output(output_dir, deadline=deadline)
        if status != 200 and not recovered:
            error_message = f"HTTP {status}: {text[:300]}"
    except Exception as error:
        error_message = str(error)
    return {
        "name": request["instances"][0].get("name"),
        "status": "FAILED" if error_message else "SUCCEEDED",
        "seconds": round(time.monotonic() - start),
        "output_dir": output_dir,
        "error": error_message,
    }


def run_batch(input_uri: str, endpoint_url: str, dry_run_only: bool = False) -> list[dict]:
    """Validates all requests in input_uri and runs the batch unless dry_run_only is True."""
    if not (endpoint_url.startswith("https://") and endpoint_url.endswith(":predict")):
        raise ValueError("--endpoint-url must be an https://...:predict URL.")
    requests = load_requests(input_uri)
    workers = replica_count(endpoint_url)

    print(
        f"Loaded {len(requests)} requests. "
        f"Running dry run across {workers} endpoint replica(s)..."
    )
    problems = dry_run(requests, endpoint_url, workers)
    if problems:
        raise RuntimeError("Dry run failed:\n  " + "\n  ".join(problems))
    print("All requests passed dry run.")
    if dry_run_only:
        return []

    results = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(run_one, req, endpoint_url) for req in requests]
        # Print each fold as it finishes so multi-GPU batches report live progress.
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            detail = row["error"] or row["output_dir"]
            status_line = f"{row['name']}: {row['status']} in {row['seconds']}s ({detail})"
            print(f"[{i}/{len(requests)}] {status_line}")
            results.append(row)
    return results


def main(argv: list[str] | None = None) -> int:
    """Parses CLI arguments and executes run_batch."""
    parser = argparse.ArgumentParser(description="Runs a batch of AlphaFold 3 predictions.")
    parser.add_argument("--input", required=True, help="A gs:// folder of .json request files.")
    parser.add_argument("--endpoint-url", required=True, help="The endpoint's full :predict URL.")
    parser.add_argument("--dry-run", action="store_true", help="Validate requests only and exit.")
    args = parser.parse_args(argv)

    try:
        rows = run_batch(args.input, args.endpoint_url, dry_run_only=args.dry_run)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 1 if any(row["status"] == "FAILED" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
