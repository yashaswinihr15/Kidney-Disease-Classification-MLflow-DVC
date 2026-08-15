import os
import shutil
import gdown
import py7zr

from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the URL.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(
                "artifacts/data_ingestion",
                exist_ok=True
            )

            logger.info(
                f"Downloading data from {dataset_url} "
                f"into file {zip_download_dir}"
            )

            file_id = dataset_url.split("/")[-2]

            prefix = (
                "https://drive.google.com/uc"
                "?export=download&id="
            )

            gdown.download(
                prefix + file_id,
                zip_download_dir
            )

            logger.info(
                f"Downloaded data from {dataset_url} "
                f"into file {zip_download_dir}"
            )

        except Exception as e:
            raise e

    def extract_zip_file(self):

        unzip_path = self.config.unzip_dir
        temp_path = os.path.join(
            unzip_path,
            "temp_extract"
        )

        # Create extraction directory
        os.makedirs(
            unzip_path,
            exist_ok=True
        )

        # Remove previous temporary extraction
        if os.path.exists(temp_path):
            shutil.rmtree(
                temp_path,
                ignore_errors=True
            )

        os.makedirs(
            temp_path,
            exist_ok=True
        )

        # -------------------------------------------------
        # Extract using 7-Zip
        # -------------------------------------------------
        with py7zr.SevenZipFile(
            self.config.local_data_file,
            mode="r"
        ) as archive:

            archive.extractall(
                path=temp_path
            )

        # -------------------------------------------------
        # Find folder containing Normal and Tumor
        # -------------------------------------------------
        dataset_source = None

        for root, dirs, files in os.walk(temp_path):

            if (
                "Normal" in dirs
                and
                "Tumor" in dirs
            ):
                dataset_source = root
                break

        if dataset_source is None:
            raise Exception(
                "Could not find Normal and Tumor "
                "folders in extracted dataset."
            )

        logger.info(
            f"Dataset folder found at: {dataset_source}"
        )

        # -------------------------------------------------
        # Final dataset folder
        # -------------------------------------------------
        expected_folder = os.path.join(
            unzip_path,
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )

        # Delete previous dataset
        if os.path.exists(expected_folder):
            shutil.rmtree(
                expected_folder,
                ignore_errors=True
            )

        os.makedirs(
            expected_folder,
            exist_ok=True
        )

        # -------------------------------------------------
        # Copy ONLY Normal and Tumor
        # -------------------------------------------------
        for class_name in [
            "Normal",
            "Tumor"
        ]:

            source_class = os.path.join(
                dataset_source,
                class_name
            )

            destination_class = os.path.join(
                expected_folder,
                class_name
            )

            if not os.path.exists(source_class):
                raise Exception(
                    f"{class_name} folder not found!"
                )

            shutil.copytree(
                source_class,
                destination_class
            )

            logger.info(
                f"Copied {class_name} dataset successfully."
            )

        # -------------------------------------------------
        # Remove temporary extraction
        # -------------------------------------------------
        if os.path.exists(temp_path):
            shutil.rmtree(
                temp_path,
                ignore_errors=True
            )

        logger.info(
            "Dataset extraction completed successfully."
        )

        logger.info(
            "ONLY Normal and Tumor are present."
        )