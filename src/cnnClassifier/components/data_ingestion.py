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
        import shutil

        unzip_path = self.config.unzip_dir
        temp_path = os.path.join(unzip_path, "temp_extract")

        # Create directories
        os.makedirs(unzip_path, exist_ok=True)

        # Remove previous incomplete extraction
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)

        os.makedirs(temp_path, exist_ok=True)

        # Extract using 7-Zip
        with py7zr.SevenZipFile(
            self.config.local_data_file,
            mode="r"
        ) as archive:
            archive.extractall(path=temp_path)

        # Find the actual dataset folder
        extracted_folder = os.path.join(
            temp_path,
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )

        expected_folder = os.path.join(
            unzip_path,
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )

        # Remove old incomplete dataset if present
        if os.path.exists(expected_folder):
            shutil.rmtree(expected_folder, ignore_errors=True)

        # Move completed extraction into final location
        if os.path.exists(extracted_folder):
            shutil.move(extracted_folder, expected_folder)

        # Remove temporary directory
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
            