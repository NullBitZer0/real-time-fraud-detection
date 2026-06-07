#!/usr/bin/env python3
"""S3 smoke test for CI — verifies DAGsHub S3 is reachable with the
token supplied via standard boto3 env vars.

Required env vars (set by the CD workflow):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_ENDPOINT_URL_S3    (e.g. https://dagshub.com/<owner>/<repo>.s3)

Exits 0 on success, prints a clear error and exits 1 on failure.
"""
import os
import sys
import boto3


def main() -> int:
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3", "")
    if not endpoint:
        print("✗ AWS_ENDPOINT_URL_S3 is not set", file=sys.stderr)
        return 1

    print(f"▶ Endpoint:        {endpoint}")
    key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    print(f"▶ Access key len:  {len(key)}  (first 8: {key[:8] or '–'})")
    print(f"▶ Secret key len:  {len(os.environ.get('AWS_SECRET_ACCESS_KEY', ''))}")

    try:
        client = boto3.client("s3", endpoint_url=endpoint)
        # A small, known object from our DVC cache (one of the smaller files).
        # This HEAD just verifies the credentials work; it doesn't need to succeed
        # for the specific key to exist.
        client.head_object(Bucket="dvc", Key="files/md5/4c/59edef30962764922a9893fb5f5cd4")
        print("  ✓ S3 HEAD OK — credentials accepted, bucket reachable")
        return 0
    except Exception as e:
        print(f"  ✗ S3 HEAD failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
