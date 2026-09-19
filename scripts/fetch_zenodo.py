#!/usr/bin/env python3
"""Download and verify the PiMorph data record from Zenodo (concept DOI 10.5281/zenodo.22839866).

    python scripts/fetch_zenodo.py --models                 # checkpoints into models/
    python scripts/fetch_zenodo.py --only pimorph_results.tar --extract     # benchmark tables and figures into runs/
    python scripts/fetch_zenodo.py --only pimorph_training_tiles.tar --extract   # tiles into data/tiles/
    python scripts/fetch_zenodo.py                          # everything into output/zenodo

The record id is looked up through the Zenodo API so the checksums come from the same version
that is downloaded. Archives extract into the original repository layout. Historical root,
docs/ and models/ Markdown is excluded by default to preserve the current consolidated
documentation; run-specific reports are retained. Use --include-archived-docs only when an
exact historical documentation restore is intended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.parse
import urllib.request

CONCEPT_DOI = os.environ.get("PIMORPH_ZENODO_CONCEPT_DOI", "10.5281/zenodo.22839866")
API = "https://zenodo.org/api/records"


def latest_record(concept_doi: str) -> dict:
    if not concept_doi:
        raise RuntimeError("no concept DOI configured yet (CONCEPT_DOI in this script or PIMORPH_ZENODO_CONCEPT_DOI)")
    query = urllib.parse.quote(f'conceptdoi:"{concept_doi}"')
    with urllib.request.urlopen(f"{API}?q={query}&size=1") as response:
        hits = json.load(response)["hits"]["hits"]
    if not hits:
        raise RuntimeError(f"no Zenodo record found for concept DOI {concept_doi}")
    return hits[0]


def md5_file(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: str) -> None:
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        for chunk in iter(lambda: response.read(8 << 20), b""):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {os.path.basename(dest)}: {done / 1e6:,.0f}/{total / 1e6:,.0f} MB", end="", flush=True)
    print()
    os.replace(tmp, dest)


def extract_archive(path: str, destination: str = ".", *, include_archived_docs: bool = False) -> tuple[int, int]:
    """Extract payloads safely without replacing current narrative documentation."""
    from pathlib import PurePosixPath

    with tarfile.open(path) as archive:
        members = archive.getmembers()
        selected = []
        for member in members:
            parts = PurePosixPath(member.name).parts
            narrative = (
                member.isfile()
                and member.name.lower().endswith(".md")
                and (len(parts) == 1 or parts[0] in {"docs", "models"})
            )
            if include_archived_docs or not narrative:
                selected.append(member)
        archive.extractall(destination, members=selected, filter="data")
    return len(selected), len(members) - len(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default="output/zenodo")
    parser.add_argument("--only", nargs="*", default=None, help="file names to fetch (default: all)")
    parser.add_argument("--extract", action="store_true", help="extract fetched tar archives into the repository root")
    parser.add_argument(
        "--include-archived-docs", action="store_true",
        help="also restore historical root/docs/models Markdown, replacing current documentation",
    )
    parser.add_argument("--models", action="store_true", help="fetch pimorph_models.tar and extract into models/")
    parser.add_argument("--concept-doi", default=CONCEPT_DOI)
    args = parser.parse_args(argv)
    if args.models:
        args.only = ["pimorph_models.tar"]
        args.extract = True
    record = latest_record(args.concept_doi)
    print(f"record {record['id']} version {record['metadata'].get('version')} doi {record.get('doi')}")
    os.makedirs(args.dest, exist_ok=True)
    for entry in record["files"]:
        name = entry["key"]
        if args.only is not None and name not in args.only:
            continue
        path = os.path.join(args.dest, name)
        expected = entry["checksum"].split(":", 1)[-1]
        if os.path.exists(path) and md5_file(path) == expected:
            print(f"  {name}: present, checksum ok")
        else:
            download(entry["links"]["self"], path)
            actual = md5_file(path)
            if actual != expected:
                raise RuntimeError(f"checksum mismatch for {name}: {actual} != {expected}")
            print(f"  {name}: downloaded, checksum ok")
        if args.extract and name.endswith(".tar"):
            extracted, skipped = extract_archive(path, include_archived_docs=args.include_archived_docs)
            print(f"  extracted {extracted} members of {name}; preserved current documentation ({skipped} excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
