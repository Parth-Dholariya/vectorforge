#!/usr/bin/env python3
"""Download the ANN benchmark corpora into data/.

    python scripts/fetch_datasets.py sift
    python scripts/fetch_datasets.py gist

SIFT-1M is the headline number: 1M x 128 SIFT descriptors with published
ground truth, and the dataset every HNSW paper and every FAISS benchmark
reports on, so results are directly comparable.
"""

import hashlib
import os
import sys
import tarfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DATASETS = {
    "sift": {
        # 1M base x 128d, 10k queries, ground truth included.
        "urls": [
            "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
            "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift.tar.gz",
        ],
        "archive": "sift.tar.gz",
        "check": "sift/sift_base.fvecs",
        "approx_mb": 161,
    },
    "gist": {
        # 1M base x 960d. 3.8 GB in RAM as float32 - this is the one that a
        # 32-bit toolchain cannot load at all.
        "urls": [
            "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        ],
        "archive": "gist.tar.gz",
        "check": "gist/gist_base.fvecs",
        "approx_mb": 2600,
    },
}


def progress(done, total, label):
    if total <= 0:
        sys.stderr.write(f"\r{label}: {done / 1e6:.1f} MB")
    else:
        pct = 100.0 * done / total
        sys.stderr.write(f"\r{label}: {pct:5.1f}%  ({done / 1e6:.1f} / {total / 1e6:.1f} MB)")
    sys.stderr.flush()


def download(urls, dest, label):
    last_err = None
    for url in urls:
        try:
            sys.stderr.write(f"trying {url}\n")
            req = urllib.request.Request(url, headers={"User-Agent": "vectorforge/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    progress(done, total, label)
            sys.stderr.write("\n")
            return True
        except Exception as e:  # noqa: BLE001 - any transport failure means try the next mirror
            last_err = e
            sys.stderr.write(f"\n  failed: {e}\n")
            if os.path.exists(dest):
                os.remove(dest)
    sys.stderr.write(f"all mirrors failed, last error: {last_err}\n")
    return False


def main():
    names = sys.argv[1:] or ["sift"]
    os.makedirs(DATA_DIR, exist_ok=True)

    for name in names:
        if name not in DATASETS:
            sys.stderr.write(f"unknown dataset '{name}'; known: {', '.join(DATASETS)}\n")
            return 1
        spec = DATASETS[name]

        marker = os.path.join(DATA_DIR, spec["check"])
        if os.path.exists(marker):
            sys.stderr.write(f"{name}: already present at {marker}\n")
            continue

        archive = os.path.join(DATA_DIR, spec["archive"])
        if not os.path.exists(archive):
            sys.stderr.write(f"{name}: downloading ~{spec['approx_mb']} MB\n")
            if not download(spec["urls"], archive, name):
                return 1

        sys.stderr.write(f"{name}: extracting\n")
        with tarfile.open(archive, "r:gz") as tar:
            # Refuse absolute paths and .. escapes rather than trusting the archive.
            for member in tar.getmembers():
                target = os.path.realpath(os.path.join(DATA_DIR, member.name))
                if not target.startswith(os.path.realpath(DATA_DIR) + os.sep):
                    sys.stderr.write(f"refusing unsafe path in archive: {member.name}\n")
                    return 1
            tar.extractall(DATA_DIR, filter="data")

        os.remove(archive)
        sys.stderr.write(f"{name}: ready at {os.path.join(DATA_DIR, name)}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
