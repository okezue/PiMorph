#!/usr/bin/env python3
"""Multipart upload of large files into a Zenodo draft (InvenioRDM records API).

A single PUT of a file above roughly 1 GB is cut off by Zenodo's proxy when the link is slow
(502 or broken pipe after about 20 minutes). The records API instead registers the file with
transfer type "M", hands out one URL per part, accepts the parts as independent requests of a
few minutes each, and assembles the file on commit. Parts are retried individually.

    ZENODO_TOKEN=... python scripts/zenodo_multipart_upload.py --record 22886559 \
        --part-mb 100 output/zenodo_v1_2/pimorph_results.tar [more files...]

The record id is the draft's id (the same number as the deposition id). Files already
present with the same size are skipped; a file present with a different size is replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://zenodo.org/api"


def _req(method, url, token, data=None, content_type=None, timeout=1800):
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if content_type:
        request.add_header("Content-Type", content_type)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        return response.status, (
            json.loads(body) if body and response.headers.get_content_type() == "application/json" else body
        )


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(record, token):
    _, d = _req("GET", f"{API}/records/{record}/draft/files", token)
    return {e["key"]: e for e in d.get("entries", [])}


def start_multipart(record, token, key, size, part_size):
    parts = (size + part_size - 1) // part_size
    payload = [{"key": key, "size": size, "transfer": {"type": "M", "parts": parts, "part_size": part_size}}]
    status, d = _req(
        "POST", f"{API}/records/{record}/draft/files", token, json.dumps(payload).encode(), "application/json"
    )
    entry = next(e for e in d["entries"] if e["key"] == key)
    return entry, parts


def put_part(url, data, token, attempts=4):
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, data=data, method="PUT")
            request.add_header("Authorization", f"Bearer {token}")
            request.add_header("Content-Type", "application/octet-stream")
            request.add_header("Content-Length", str(len(data)))
            with urllib.request.urlopen(request, timeout=1800) as response:
                return response.status
        except (urllib.error.URLError, OSError) as exc:
            print(f"    part attempt {attempt} failed: {str(exc)[:120]}", flush=True)
            time.sleep(30 * attempt)
    raise RuntimeError("part upload failed after retries")


def upload(record, token, path, part_size):
    key = os.path.basename(path)
    size = os.path.getsize(path)
    existing = list_files(record, token)
    if key in existing and existing[key].get("size") == size and existing[key].get("status") == "completed":
        print(f"{key}: already in the draft with the same size, skipped")
        return
    if key in existing:
        print(f"{key}: replacing the existing entry")
        _req("DELETE", f"{API}/records/{record}/draft/files/{key}", token)
    entry, n_parts = start_multipart(record, token, key, size, part_size)
    links = entry["links"]
    part_urls = {int(p["part"]): p["url"] for p in links["parts"]} if "parts" in links else None
    print(f"{key}: {size / 1e9:.2f} GB in {n_parts} parts of {part_size >> 20} MB", flush=True)
    t0 = time.time()
    with open(path, "rb") as f:
        for i in range(1, n_parts + 1):
            data = f.read(part_size)
            url = part_urls[i] if part_urls else f"{links['content']}?part={i}"
            put_part(url, data, token)
            done = min(i * part_size, size)
            rate = done / max(time.time() - t0, 1e-6) / 1e6
            print(f"  part {i}/{n_parts} ok  {done / 1e9:.2f}/{size / 1e9:.2f} GB  {rate:.2f} MB/s", flush=True)
    status, d = _req("POST", links["commit"], token)
    checksum = d.get("checksum", "") if isinstance(d, dict) else ""
    local = md5_file(path)
    remote = checksum.split(":", 1)[-1]
    if remote and remote != local:
        raise RuntimeError(f"{key}: checksum mismatch after commit: {remote} != {local}")
    print(f"{key}: committed, md5 {local}{' (verified)' if remote else ''}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", required=True, help="draft record id")
    ap.add_argument("--part-mb", type=int, default=100)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ZENODO_TOKEN is not set", file=sys.stderr)
        return 2
    for path in args.files:
        upload(args.record, token, path, args.part_mb << 20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
