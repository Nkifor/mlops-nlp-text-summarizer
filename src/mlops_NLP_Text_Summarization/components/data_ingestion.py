import os
import urllib.request as request
import zipfile
import pandas as pd
from datasets import load_dataset
import json
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
            os.makedirs(self.config.local_data_dir, exist_ok=True)
            dataset = load_dataset('samsum')
            dataset.save_to_disk(self.config.local_data_dir)
            for split in ['train', 'test', 'validation']:
                os.makedirs(self.config.local_data_dir + f"/{split}", exist_ok=True)
                dataset[split].to_csv(self.config.local_data_dir + f"/{split}/{split}.csv")
            train_json = dataset['train'].to_json(self.config.local_data_dir  + "/train.json")
            test_json= dataset['test'].to_json(self.config.local_data_dir + "/test.json")
            validation_json= dataset['validation'].to_json(self.config.local_data_dir  + "/validation.json")
            with open(self.config.local_data_dir + '/dataset_dict.json', 'w') as f:
                json.dump({'train': train_json, 'test': test_json, 'validation': validation_json}, f)
        else:
            logger.info(f"Folder already exists of size: {get_size(Path(self.config.local_data_dir))}")