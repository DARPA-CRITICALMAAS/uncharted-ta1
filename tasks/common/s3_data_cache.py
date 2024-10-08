import logging, os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

from typing import Optional

from mypy_boto3_s3 import ServiceResource
from mypy_boto3_s3.service_resource import Bucket

logger = logging.getLogger(__name__)


class S3DataCache:
    """
    Class to handle downloading data from S3
    and caching to local disk
    """

    def __init__(
        self,
        local_cache_path: str,
        s3_url: str = "",
        s3_bucket: str = "lara",
        aws_region_name: str = "",
        aws_access_key_id: str = "",
        aws_secret_access_key="",
    ):
        # s3 connection params
        self.s3_url = s3_url
        self.s3_bucket = s3_bucket
        self.aws_region_name = aws_region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_resource: Optional[ServiceResource] = None

        if not local_cache_path:
            raise Exception("local_cache_path not given for S3ModelCache")

        # create local cache dirs, if needed
        self.local_cache_path = local_cache_path
        if not os.path.exists(self.local_cache_path):
            os.makedirs(self.local_cache_path)

    def _s3_resource(self) -> ServiceResource:
        """lazily initialize the s3 resource"""
        if not self.s3_resource:
            self._init_s3(
                self.s3_url,
                self.s3_bucket,
                self.aws_region_name,
                self.aws_access_key_id,
                self.aws_secret_access_key,
            )
        if not self.s3_resource:
            raise Exception("S3 resource not initialized")
        return self.s3_resource

    def _init_s3(
        self,
        s3_url,
        s3_bucket,
        aws_region_name,
        aws_access_key_id,
        aws_secret_access_key,
    ):
        """connect to s3"""
        logger.info(f"Connecting to S3 at {s3_url}")
        self.s3_resource = boto3.resource(
            service_name="s3",
            endpoint_url=s3_url if s3_url != "" else None,
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        if self.s3_bucket:
            if not S3DataCache.bucket_exists(self.s3_resource, self.s3_bucket):
                raise Exception("S3 bucket {} does not exist".format(s3_bucket))
        else:
            logger.warning("S3 bucket name not given! Access to S3 not possible")

    def fetch_file_from_s3(self, s3_obj_key: str, overwrite: bool = False) -> str:
        """
        Fetch file, S3 -> local disk
        (Overwrite is False by default)
        Returns local path where file is stored
        """

        if not self.s3_bucket:
            logger.warning(
                f"Skipping fetch_data_from_s3 {s3_obj_key}. No s3_bucket given"
            )
            return ""

        # local path to store file
        filepath_local = os.path.join(self.local_cache_path, s3_obj_key)
        if Path(filepath_local).exists() and not overwrite:
            logger.info(
                f"Data already exists on local path {filepath_local}. Skipping data download from S3."
            )
            return filepath_local

        # create local dirs, if needed
        os.makedirs(filepath_local[: filepath_local.rfind("/")], exist_ok=True)

        logger.info(f"Downloading data from s3 to local disk: {s3_obj_key}")
        s3_resource = self._s3_resource()
        bucket = s3_resource.Bucket(self.s3_bucket)
        S3DataCache.get_s3_object_to_file(bucket, s3_obj_key, filepath_local)

        return filepath_local

    def list_bucket_contents(self, path_prefix: str):
        """
        List an S3 bucket contents based on key prefix
        Returns list of matching keys
        """
        resp = self._s3_resource().meta.client.list_objects_v2(  # type: ignore
            Bucket=self.s3_bucket, Prefix=path_prefix, MaxKeys=1000
        )
        return [obj["Key"] for obj in resp.get("Contents", []) if "Key" in obj]

    @staticmethod
    def bucket_exists(s3_resource: ServiceResource, bucket_name: str) -> bool:
        """
        Check if an s3 bucket exists
        """
        exists = False
        try:
            # check if bucket exists

            s3_resource.meta.client.head_bucket(Bucket=bucket_name)  # type: ignore
            logger.debug("Bucket {} exists.".format(bucket_name))
            exists = True
        except ClientError as e:
            logger.error(
                "Bucket {} doesn't exist or you don't have access to it.".format(
                    bucket_name
                )
            )
        return exists

    @staticmethod
    def get_s3_object_to_file(bucket: Bucket, object_key: str, filename: str):
        """
        Download an object from an s3 bucket and save to local file
        """
        try:
            bucket.download_file(object_key, filename)
            logger.debug(
                "Got object {} from bucket {} and saved to {}".format(
                    object_key, bucket.name, filename
                )
            )
        except ClientError as e:
            logger.exception(
                "Error! Couldn't get and save object {} from bucket {}".format(
                    object_key, bucket.name
                ),
                exc_info=True,
            )
            raise
