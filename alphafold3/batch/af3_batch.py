#!/usr/bin/env python3
"""Runs a batch of AlphaFold 3 predictions on Google Cloud.

Validates every request file in a Cloud Storage folder with an upfront dry run, then runs the
predictions in parallel across the endpoint's GPU replicas.
"""

import argparse
from collections.abc import Callable
import concurrent.futures
import copy
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

GCS_API = "https://storage.googleapis.com/storage/v1"
METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
)
# Dedicated endpoints stop a fold after one hour, bounding all request and polling waits.
FOLD_TIMEOUT_SEC = 3700

_token_lock = threading.Lock()
_token_cache: dict = {"value": "", "expires": 0.0}


def access_token() -> str:
    """Returns a cached OAuth access token from Cloud Run metadata or the gcloud CLI."""
    with _token_lock:
        if time.time() < _token_cache["expires"]:
            return _token_cache["value"]
        detail = ""
        for delay in (1, 2, 4, 8, 0):
            try:
                if os.environ.get("CLOUD_RUN_JOB"):
                    req = urllib.request.Request(
                        METADATA_URL, headers={"Metadata-Flavor": "Google"}
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        token = str(json.loads(resp.read()).get("access_token", "")).strip()
                else:
                    token = subprocess.check_output(
                        ["gcloud", "auth", "print-access-token"],
                        stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=15,
                    ).strip()
                if not token:
                    raise ValueError("Empty access token returned.")
                # Issued tokens have >=5 minutes remaining; wall-clock caching survives sleep.
                _token_cache.update(value=token, expires=time.time() + 240)
                return token
            except urllib.error.HTTPError as error:
                with error:
                    detail = f"HTTP {error.code}: {error.read().decode(errors='replace').strip()}"
            except (
                OSError, http.client.HTTPException, subprocess.SubprocessError, ValueError
            ) as error:
                detail = (getattr(error, "stderr", "") or str(error)).strip()
            if delay:
                time.sleep(delay)
        raise RuntimeError(f"Failed to get access token: {detail}")


def call(url: str, data: bytes | None = None, timeout: float = 60) -> tuple[int, str]:
    """Sends an authenticated GET, or a POST when data is set, and returns (status, text)."""
    delays = (1, 2, 4, 8, 0) if data is None else (0,)
    status, text = 0, ""
    for delay in delays:
        headers = {"Authorization": f"Bearer {access_token()}", "Content-Type": "application/json"}
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data, headers), timeout=timeout
            ) as resp:
                return resp.status, resp.read().decode(errors="replace")
        except urllib.error.HTTPError as error:
            with error:
                status, text = error.code, error.read().decode(errors="replace")
        except (OSError, http.client.HTTPException) as error:
            status, text = 0, str(error)
        if status == 401:
            _token_cache["expires"] = 0.0
        if not delay or status not in (0, 429, 500, 502, 503, 504):
            break
        time.sleep(delay)
    return status, text


def gcs_split(uri: object) -> tuple[str, str]:
    """Splits gs://<bucket>/<path> into (bucket, trimmed_path) or raises ValueError."""
    raw = uri.removeprefix("gs://").rstrip("/") if isinstance(uri, str) else ""
    bucket, _, path = raw.partition("/")
    if not (isinstance(uri, str) and uri.startswith("gs://") and bucket and path
            and "//" not in f"/{path}"):
        raise ValueError(f"Expected gs://<bucket>/<folder>, got {uri!r}.")
    return bucket, path


def pool_map(fn: Callable[..., object], items: list, workers: int) -> list:
    """Maps fn across items in a thread pool, cancelling pending work immediately on interrupt."""
    pool = concurrent.futures.ThreadPoolExecutor(min(workers, len(items)))
    try:
        return list(pool.map(fn, items))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def gcs_list(folder_uri: str, limit: int = 0, flat: bool = False) -> list[str]:
    """Returns gs:// URIs of objects under a gs://<bucket>/<folder> prefix."""
    bucket, prefix = gcs_split(folder_uri)
    # Request only object names and pagination tokens to avoid downloading full object metadata.
    query: dict = {"prefix": prefix + "/", "fields": "items/name,nextPageToken"}
    if limit:
        query["maxResults"] = limit
    if flat:
        query["delimiter"] = "/"
    uris: list[str] = []
    while True:
        status, text = call(f"{GCS_API}/b/{bucket}/o?{urllib.parse.urlencode(query)}")
        if status != 200:
            raise RuntimeError(f"Listing {folder_uri} failed with HTTP {status}: {text}")
        page = json.loads(text)
        uris.extend(f"gs://{bucket}/{item['name']}" for item in page.get("items", []))
        if limit or "nextPageToken" not in page:
            return uris
        query["pageToken"] = page["nextPageToken"]


def read_json(uri: str) -> tuple[str, object]:
    """Downloads and parses one JSON request object from Cloud Storage."""
    bucket, name = gcs_split(uri)
    status, text = call(f"{GCS_API}/b/{bucket}/o/{urllib.parse.quote(name, safe='')}?alt=media")
    if status != 200:
        raise RuntimeError(f"Reading {uri} failed with HTTP {status}: {text}")
    try:
        return uri, json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {uri}: {error}") from error


def load_requests(input_uri: str) -> list[tuple[str, object]]:
    """Loads top-level .json prediction requests from a gs://<bucket>/<folder>."""
    uris = [uri for uri in sorted(gcs_list(input_uri, flat=True)) if uri.endswith(".json")]
    if not uris:
        raise ValueError(f"No .json request files found in {input_uri}.")
    return pool_map(read_json, uris, 16)


def replica_count(endpoint_url: str) -> int:
    """Returns the endpoint's active replica count from the control plane, or 1 if unavailable."""
    path = urllib.parse.urlsplit(endpoint_url).path.removesuffix(":predict")
    parts = path.split("/")
    if len(parts) < 6 or parts[4] != "locations":
        return 1
    # Endpoint metadata sits on the regional aiplatform.googleapis.com host even for dedicated URLs.
    status, text = call(f"https://{parts[5]}-aiplatform.googleapis.com{path}", timeout=5)
    if status != 200:
        return 1
    endpoint = json.loads(text)
    traffic = endpoint.get("trafficSplit", {})
    replicas = 0
    for model in endpoint.get("deployedModels", []):
        if traffic.get(model.get("id")):
            res = model.get("dedicatedResources", {})
            replicas += res.get("maxReplicaCount") or res.get("minReplicaCount") or 0
    return replicas or 1


def is_complete(uris: list[str], output_dir: str, inference: bool) -> bool:
    """Returns True when uris contain a finished AlphaFold 3 output set under output_dir."""
    bucket, path = gcs_split(output_dir)
    prefix = f"gs://{bucket}/{path}/"
    # Top-ranked outputs sit at output_dir/<job>/<file>; per-sample copies sit one level deeper.
    top = [u[len(prefix):].removesuffix(".zst") for u in uris if u[len(prefix):].count("/") == 1]
    if not inference:
        return any(f.endswith("_data.json") for f in top)
    need = ("_summary_confidences.json", "_model.cif", "_ranking_scores.csv", "_data.json")
    has_pae = any(f.endswith("_confidences.json") and "_summary_" not in f for f in top)
    return has_pae and all(any(f.endswith(s) for f in top) for s in need)


def has_output(output_dir: str, deadline: float, inference: bool = True) -> bool:
    """Polls output_dir every 30s until deadline for completed AlphaFold 3 output files."""
    while True:
        try:
            if is_complete(gcs_list(output_dir), output_dir, inference):
                return True
        except (RuntimeError, ValueError) as error:
            print(f"Retrying output check for {output_dir}: {error}", flush=True)
        if time.monotonic() >= deadline:
            return False
        time.sleep(30)


def predict(endpoint_url: str, request: dict, timeout: float) -> tuple[int, str, float]:
    """Posts one request, retrying with jitter while all replicas are busy (HTTP 429)."""
    # Encode the JSON payload once so HTTP 429 retries do not re-serialize large MSA inputs.
    data = json.dumps(request).encode()
    give_up_at = time.monotonic() + FOLD_TIMEOUT_SEC
    while True:
        sent_at = time.monotonic()
        status, text = call(endpoint_url, data, timeout)
        if status != 429 or sent_at > give_up_at:
            return status, text, sent_at
        # Random jitter prevents multi-GPU worker threads from colliding in lockstep on Envoy.
        time.sleep(random.uniform(0.3, 1.0))


def check_output(args: tuple[tuple[str, dict], bool]) -> tuple[bool, str | None]:
    """Checks one output_dir in Cloud Storage, returning (is_done, error_or_None)."""
    (uri, req), resume = args
    label = req["instances"][0].get("name") or uri.rsplit("/", 1)[-1]
    params = req.get("parameters") or req["instances"][0]["parameters"]
    output_dir = str(params["output_dir"]).rstrip("/")
    try:
        uris = gcs_list(output_dir, limit=0 if resume else 1)
    except RuntimeError as error:
        return False, f"{label} ({uri}): {error}"
    if not uris:
        return False, None
    if resume and is_complete(uris, output_dir, params.get("run_inference") is not False):
        return True, None
    return False, f"{label} ({uri}): output_dir {output_dir} is not empty."


def check_one(args: tuple[tuple[str, dict], str]) -> str | None:
    """Runs server-side dry_run validation on a copy of one request."""
    (uri, req), endpoint_url = args
    label = req["instances"][0].get("name") or uri.rsplit("/", 1)[-1]
    payload = copy.deepcopy(req)
    (payload.get("parameters") or payload["instances"][0]["parameters"])["dry_run"] = True
    status, text, _ = predict(endpoint_url, payload, timeout=120)
    return None if status == 200 else f"{label} ({uri}): dry run failed with HTTP {status}: {text}"


def dry_run(
    items: list[tuple[str, object]], input_uri: str, endpoint_url: str, resume: bool
) -> tuple[list[tuple[str, dict]], int]:
    """Validates schemas, output folders, and server dry_run; returns (pending_items, workers)."""
    in_bucket, in_path = gcs_split(input_uri)
    dirs = [(f"gs://{in_bucket}/{in_path}/", "--input")]
    valid: list[tuple[str, dict]] = []
    problems: list[str] = []
    for uri, req in items:
        inst = req.get("instances") if isinstance(req, dict) else None
        if not (isinstance(inst, list) and len(inst) == 1 and isinstance(inst[0], dict)):
            problems.append(f"{uri}: 'instances' must be a list with one AlphaFold 3 input object.")
            continue
        params = req.get("parameters") or inst[0].get("parameters")
        if not isinstance(params, dict):
            problems.append(f"{uri}: 'parameters' must be a JSON object with 'output_dir'.")
            continue
        try:
            out_bucket, out_path = gcs_split(params.get("output_dir"))
            dirs.append((f"gs://{out_bucket}/{out_path}/", uri))
            valid.append((uri, req))
        except ValueError as error:
            problems.append(f"{uri}: output_dir: {error}")
    # Keep outer fixed while nested children match so multiple overlaps under one folder surface.
    dirs.sort()
    outer, outer_uri = dirs[0]
    for inner, inner_uri in dirs[1:]:
        if inner.startswith(outer):
            problems.append(f"{inner_uri}: {inner[:-1]} overlaps {outer[:-1]} ({outer_uri}).")
        else:
            outer, outer_uri = inner, inner_uri
    if problems:
        raise RuntimeError("Dry run failed:\n  " + "\n  ".join(problems))

    pending: list[tuple[str, dict]] = []
    for item, (done, err) in zip(valid, pool_map(check_output, [(x, resume) for x in valid], 16)):
        if err:
            problems.append(err)
        elif not done:
            pending.append(item)
    if problems:
        raise RuntimeError("Dry run failed:\n  " + "\n  ".join(problems))
    if resume:
        skipped = len(valid) - len(pending)
        print(f"Skipped {skipped} completed request(s); {len(pending)} remaining.", flush=True)
    if not pending:
        return [], 1

    workers = replica_count(endpoint_url)
    print(f"Running dry run on {len(pending)} request(s) across {workers} replicas...", flush=True)
    problems = [p for p in pool_map(check_one, [(x, endpoint_url) for x in pending], workers) if p]
    if problems:
        raise RuntimeError("Dry run failed:\n  " + "\n  ".join(problems))
    print("All requests passed dry run.", flush=True)
    return pending, workers


def run_one(item: tuple[str, dict], endpoint_url: str) -> dict:
    """Runs one request and returns its status, elapsed seconds, output folder, and error if any."""
    uri, req = item
    params = req.get("parameters") or req["instances"][0]["parameters"]
    output_dir = str(params["output_dir"]).rstrip("/")
    inference = params.get("run_inference") is not False
    start = time.monotonic()
    status, text, sent_at = predict(endpoint_url, req, FOLD_TIMEOUT_SEC)
    # Long folds can outlast intermediate HTTP idle timeouts while finishing on the endpoint.
    dropped = status in (0, 502, 503, 504) and (time.monotonic() - sent_at) >= 10
    recovered = dropped and has_output(output_dir, sent_at + FOLD_TIMEOUT_SEC, inference)
    error_message = "" if (status == 200 or recovered) else f"HTTP {status}: {text}"
    return {
        "name": req["instances"][0].get("name") or uri.rsplit("/", 1)[-1],
        "status": "FAILED" if error_message else "SUCCEEDED",
        "seconds": round(time.monotonic() - start),
        "output_dir": output_dir,
        "error": error_message,
    }


def run_batch(
    input_uri: str, endpoint_url: str, dry_run_only: bool = False, resume: bool = False
) -> list[dict]:
    """Validates all requests in input_uri and runs the batch unless dry_run_only is True."""
    if not (endpoint_url.startswith("https://") and endpoint_url.endswith(":predict")):
        raise ValueError("--endpoint-url must be an https://...:predict URL.")
    items, workers = dry_run(load_requests(input_uri), input_uri, endpoint_url, resume)
    if not items or dry_run_only:
        return []

    # Release items after submission so finished request payloads are freed during long batches.
    total = len(items)
    results: list[dict] = []
    pool = concurrent.futures.ThreadPoolExecutor(workers)
    try:
        futures = [pool.submit(run_one, item, endpoint_url) for item in items]
        items.clear()
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            detail = row["error"] or row["output_dir"]
            summary = f"{row['name']}: {row['status']} in {row['seconds']}s ({detail})"
            print(f"[{i}/{total}] {summary}", flush=True)
            results.append(row)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    succeeded = sum(row["status"] == "SUCCEEDED" for row in results)
    print(f"Batch finished: {succeeded} succeeded, {len(results) - succeeded} failed.", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    """Parses CLI arguments and executes run_batch."""
    parser = argparse.ArgumentParser(description="Runs a batch of AlphaFold 3 predictions.")
    parser.add_argument("--input", required=True, help="A gs:// folder of .json request files.")
    parser.add_argument("--endpoint-url", required=True, help="The endpoint's full :predict URL.")
    parser.add_argument("--dry-run", action="store_true", help="Validate requests only and exit.")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed requests.")
    args = parser.parse_args(argv)

    try:
        rows = run_batch(
            args.input, args.endpoint_url, dry_run_only=args.dry_run, resume=args.resume
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr, flush=True)
        return 130
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    return 1 if any(row["status"] == "FAILED" for row in rows) else 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    os._exit(exit_code) if exit_code == 130 else sys.exit(exit_code)
