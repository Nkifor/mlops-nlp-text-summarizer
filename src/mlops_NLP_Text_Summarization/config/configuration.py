from mlops_NLP_Text_Summarization.constants import *
from mlops_NLP_Text_Summarization.utils.common import read_yaml, create_directories

from mlops_NLP_Text_Summarization.entity import (DataIngestionConfigLibrary,
                                                 DataIngestionConfigLink,
                                                 DataIngestionConfigUnzipLink,
                                                 DataValidationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def choose_type_of_data_ingestion(self):
        try:
            if self.config.rulesingestion.data_ingestion_link_zipped.source_URL_zipped is not False:
                return self.get_data_ingestion_config_unzip_link()
            if self.config.rulesingestion.data_ingestion_link.source_URL is not False:
                return self.get_data_ingestion_config_link()
            if self.config.rulesingestion.data_ingestion_library_hugging_face_dataset.library_dataset_name is not False:
                return self.get_data_ingestion_config_library()
            else:
                raise ValueError("data ingestion type not supported")
        except Exception as e:
            print(e)
            raise Exception(e)




    def get_data_ingestion_config_unzip_link(self) -> DataIngestionConfigUnzipLink:
        config = self.config.rulesingestion.data_ingestion_link_zipped

        create_directories([config.root_dir])

        data_ingestion_unzip_link_config = DataIngestionConfigUnzipLink(
            root_dir=config.root_dir,
            source_URL_zipped=config.source_URL_zipped,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_unzip_link_config

    def get_data_ingestion_config_link(self) -> DataIngestionConfigLink:
        config = self.config.rulesingestion.data_ingestion_link

        create_directories([config.root_dir])

        data_ingestion_link_config = DataIngestionConfigLink(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
        )
        return data_ingestion_link_config

    def get_data_ingestion_config_library(self) -> DataIngestionConfigLibrary:
        config = self.config.rulesingestion.data_ingestion_library_hugging_face_dataset

        create_directories([config.root_dir])

        data_ingestion_library_config = DataIngestionConfigLibrary(
            root_dir=config.root_dir,
            local_data_dir=config.local_data_dir,
            library_dataset_name=config.library_dataset_name,
        )
        return data_ingestion_library_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            all_required_files_in_main_catalog=config.all_required_files_in_main_catalog,
            library_dataset_name=config.library_dataset_name,
            all_required_files_in_train_test_validations_catalogs=config.all_required_files_in_train_test_validations_catalogs
        )

        return data_validation_config