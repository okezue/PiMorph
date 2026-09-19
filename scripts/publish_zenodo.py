#!/usr/bin/env python3
"""Create or version the PiMorph data record on Zenodo.

    export ZENODO_TOKEN=...      # personal access token with deposit:write and deposit:actions
    python scripts/publish_zenodo.py --new-record --version 1.1.0 \
        --file output/zenodo_v1/pimorph_models.tar \
        --file output/zenodo_v1/pimorph_results.tar \
        --file output/zenodo_v1/pimorph_training_tiles.tar \
        --file output/zenodo_v1/README.md \
        --description-file output/zenodo_v1/description.html [--publish]

Later versions omit ``--new-record`` and version the concept DOI in CONCEPT_DOI. Without
``--publish`` the deposit is left as a draft for review in the Zenodo web UI. Files already
in the previous version are carried over unchanged; a file given here with the same name as
an existing one replaces it. The token is read from the environment only and is never
written anywhere.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

# concept DOI of the PiMorph data record (always resolves to the newest version)
CONCEPT_DOI = os.environ.get("PIMORPH_ZENODO_CONCEPT_DOI", "10.5281/zenodo.22839866")
API = "https://zenodo.org/api"
TITLE = "PiMorph: proposal checkpoints, benchmark results and training tiles for endothelial cell complexes"
CREATORS = [{"name": "Bell, Alexander Okezue", "affiliation": "Stanford University", "orcid": "0009-0004-9419-3962"}]
KEYWORDS = [
    "endothelial cells",
    "cell segmentation",
    "cell complex",
    "multicellular vertices",
    "tricellular junctions",
    "VE-cadherin",
    "adherens junctions",
    "shear stress",
    "Cellpose",
    "uncertainty quantification",
]
RELATED = [
    {"relation": "isSupplementTo", "identifier": "https://github.com/okezue/PiMorph", "resource_type": "software"},
]


def _request(method: str, url: str, token: str, data: bytes | None = None, content_type: str | None = None) -> dict:
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if content_type:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail[:500]}") from exc


def latest_record_id(concept_doi: str) -> int:
    query = urllib.parse.quote(f'conceptdoi:"{concept_doi}"')
    with urllib.request.urlopen(f"{API}/records?q={query}&size=1") as response:
        hits = json.load(response)["hits"]["hits"]
    if not hits:
        raise RuntimeError(f"no record for concept DOI {concept_doi}")
    return int(hits[0]["id"])


def md5_file(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_file(bucket_url: str, path: str, token: str) -> dict:
    name = os.path.basename(path)
    size = os.path.getsize(path)
    print(f"  uploading {name} ({size / 1e9:.2f} GB)...", flush=True)

    class Reader:
        def __init__(self, handle):
            self.handle = handle
            self.sent = 0

        def read(self, n=-1):
            chunk = self.handle.read(n)
            self.sent += len(chunk)
            if size and self.sent % (256 << 20) < len(chunk):
                print(f"\r    {self.sent / 1e9:.2f}/{size / 1e9:.2f} GB", end="", flush=True)
            return chunk

        def __len__(self):
            return size

    with open(path, "rb") as handle:
        request = urllib.request.Request(f"{bucket_url}/{urllib.parse.quote(name)}", data=Reader(handle), method="PUT")
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("Content-Type", "application/octet-stream")
        request.add_header("Content-Length", str(size))
        with urllib.request.urlopen(request) as response:
            info = json.loads(response.read())
    print()
    remote = info.get("checksum", "").split(":", 1)[-1]
    local = md5_file(path)
    if remote and remote != local:
        raise RuntimeError(f"checksum mismatch after upload of {name}: {remote} != {local}")
    print(f"    ok, md5 {local}")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="version label, e.g. 1.1.0")
    parser.add_argument("--file", action="append", default=[], help="file to add (repeatable)")
    parser.add_argument("--description-file", default=None, help="HTML file for the record description")
    parser.add_argument("--publish", action="store_true", help="publish instead of leaving a draft")
    parser.add_argument("--new-record", action="store_true", help="create the record (first version)")
    parser.add_argument("--record-id", type=int, default=None, help="record to version (default: latest)")
    parser.add_argument("--draft-id", type=int, default=None, help="reuse an existing draft deposition")
    parser.add_argument("--remove-file", action="append", default=[], help="file name to delete from the draft")
    args = parser.parse_args(argv)
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ZENODO_TOKEN is not set", file=sys.stderr)
        return 2
    for path in args.file:
        if not os.path.isfile(path):
            print(f"missing file: {path}", file=sys.stderr)
            return 2

    if args.draft_id:
        draft = _request("GET", f"{API}/deposit/depositions/{args.draft_id}", token)
        if draft.get("submitted"):
            raise RuntimeError(f"deposition {args.draft_id} is already published")
    elif args.new_record:
        if not args.description_file:
            raise RuntimeError("--new-record requires --description-file")
        print(f"creating record: {TITLE}")
        draft = _request("POST", f"{API}/deposit/depositions", token, data=b"{}", content_type="application/json")
        draft["metadata"] = {
            "upload_type": "dataset",
            "title": TITLE,
            "creators": CREATORS,
            "license": "cc-by-4.0",
            "access_right": "open",
            "keywords": KEYWORDS,
            "related_identifiers": RELATED,
        }
    else:
        if not (args.record_id or CONCEPT_DOI):
            raise RuntimeError("no CONCEPT_DOI yet: use --new-record or --record-id")
        record_id = args.record_id or latest_record_id(CONCEPT_DOI)
        print(f"creating new version of record {record_id}")
        created = _request("POST", f"{API}/deposit/depositions/{record_id}/actions/newversion", token)
        draft = _request("GET", created["links"]["latest_draft"], token)
    draft_id = draft["id"]
    bucket = draft["links"]["bucket"]
    print(f"draft deposition {draft_id}")

    existing = {f["filename"]: f for f in draft.get("files", [])}
    for name in args.remove_file:
        if name in existing:
            print(f"  removing {name}")
            _request("DELETE", f"{API}/deposit/depositions/{draft_id}/files/{existing[name]['id']}", token)
            del existing[name]
    for path in args.file:
        name = os.path.basename(path)
        if name in existing:
            print(f"  removing previous {name}")
            _request("DELETE", f"{API}/deposit/depositions/{draft_id}/files/{existing[name]['id']}", token)
        upload_file(bucket, path, token)

    metadata = dict(draft["metadata"])
    metadata["version"] = args.version
    metadata["publication_date"] = datetime.date.today().isoformat()
    if args.description_file:
        with open(args.description_file) as handle:
            metadata["description"] = handle.read()
    for key in ("doi", "prereserve_doi"):
        metadata.pop(key, None)
    _request(
        "PUT",
        f"{API}/deposit/depositions/{draft_id}",
        token,
        data=json.dumps({"metadata": metadata}).encode(),
        content_type="application/json",
    )
    print(f"metadata updated: version {args.version}")

    if args.publish:
        published = _request("POST", f"{API}/deposit/depositions/{draft_id}/actions/publish", token)
        print(f"published: {published.get('doi_url') or published.get('doi')}")
        print(f"concept DOI: {published.get('conceptdoi')}  (put this in scripts/fetch_zenodo.py and models/README.md)")
    else:
        print(f"draft left unpublished: https://zenodo.org/uploads/{draft_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
