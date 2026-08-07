import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import (DataIngestionConfig)



import os
import py7zr



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the url
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            logger.info(
                f"Downloading data from {dataset_url} into file {zip_download_dir}"
            )

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?export=download&id="

            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(
                f"Downloaded data from {dataset_url} into file {zip_download_dir}"
            )

        except Exception as e:
            raise e

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with py7zr.SevenZipFile(self.config.local_data_file, mode="r") as archive:
            archive.extractall(path=unzip_path)
            