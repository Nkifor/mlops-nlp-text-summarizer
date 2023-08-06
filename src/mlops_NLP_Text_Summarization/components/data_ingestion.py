import os
import urllib.request as request
import zipfile
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from mlops_NLP_Text_Summarization.logging import logger
from mlops_NLP_Text_Summarization.utils.common import get_size
from mlops_NLP_Text_Summarization.entity import (DataIngestionConfigLibrary,
                                                    DataIngestionConfigLink,
                                                    DataIngestionConfigUnzipLink)


class DataIngestionUnzippedLink:
    def __init__(self, config: DataIngestionConfigUnzipLink):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL_zipped,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

class DataIngestionLink:
    def __init__(self, config: DataIngestionConfigLink):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")


class DataIngestionLibrary:
    def __init__(self, config: DataIngestionConfigLibrary):
        self.config = config

    def get_data_from_library(self):
        if not os.path.exists(self.config.local_data_dir):
            directory =  os.makedirs(self.config.local_data_dir, exist_ok=True)
            directory_train = os.makedirs(self.config.local_data_dir + "/train", exist_ok=True)
            directory_test = os.makedirs(self.config.local_data_dir + "/test", exist_ok=True)
            directory_validation = os.makedirs(self.config.local_data_dir + "/validation", exist_ok=True)
            load_train = load_dataset("samsum", split ="train")
            df_load_train = pd.DataFrame(load_train)
            df_load_train.to_csv(self.config.local_data_dir +  "/train/train.csv")
            load_test = load_dataset("samsum", split ="test")
            df_load_test = pd.DataFrame(load_test)
            df_load_test.to_csv(self.config.local_data_dir + "/test/test.csv")
            load_validation = load_dataset("samsum", split ="validation")
            df_load_validation = pd.DataFrame(load_validation)
            df_load_validation.to_csv(self.config.local_data_dir + "/validation/validation.csv")

            #dataset_test = load_dataset("samsum", split ="test").to_csv(self.config.local_data_dir/"test.csv"),
            #dataset_validation = load_dataset("samsum", split ="validation").to_csv(self.config.local_data_dir/"validation.csv")

            logger.info(f" Directory in path {self.config.local_data_dir} was created from library! with following files: \n train.csv \n  test.csv \n validation.csv ")
        else:
            logger.info(f"Folder already exists of size: {get_size(Path(self.config.local_data_dir))}")