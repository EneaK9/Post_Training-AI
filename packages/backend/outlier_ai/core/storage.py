"""Object storage for renders, brand assets, snapshots, and checkpoints.

`LocalStorage` writes under a directory (dev and tests). `S3Storage` targets any
S3-compatible endpoint (MinIO locally, S3 or R2 in the cloud). Keys are stored in the
database as `storage://<key>` URIs so the backend can be swapped without a migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from outlier_ai.core.errors import ConfigError
from outlier_ai.core.settings import Settings, get_settings

URI_PREFIX = "storage://"


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def public_url(self, key: str) -> str: ...


def key_from_uri(uri: str) -> str:
    if not uri.startswith(URI_PREFIX):
        raise ValueError(f"not a storage uri: {uri}")
    return uri[len(URI_PREFIX) :]


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError("key escapes storage root")
        return p

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return URI_PREFIX + key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def public_url(self, key: str) -> str:
        return self._path(key).as_uri()


class S3Storage:
    def __init__(
        self,
        bucket: str,
        endpoint: str | None,
        access_key: str | None,
        secret_key: str | None,
        region: str,
    ) -> None:
        import boto3  # imported lazily so tests without boto3 credentials stay fast

        self.bucket = bucket
        self.endpoint = endpoint
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return URI_PREFIX + key

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.client.exceptions.ClientError:
            return False

    def public_url(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=3600
        )


def get_storage(settings: Settings | None = None) -> Storage:
    s = settings or get_settings()
    if s.storage_backend == "local":
        return LocalStorage(s.local_storage_dir)
    if not (s.s3_access_key and s.s3_secret_key):
        raise ConfigError("S3 storage selected but S3_ACCESS_KEY / S3_SECRET_KEY are missing")
    return S3Storage(s.s3_bucket, s.s3_endpoint, s.s3_access_key, s.s3_secret_key, s.s3_region)
