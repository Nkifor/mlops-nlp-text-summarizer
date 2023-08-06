from mlops_NLP_Text_Summarization.config.configuration import ConfigurationManager
from mlops_NLP_Text_Summarization.components.data_ingestion import DataIngestionLink, DataIngestionUnzippedLink, DataIngestionLibrary
from mlops_NLP_Text_Summarization.logging import logger


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        logger.info(f"ConfigManager initialized: {config}")
        data_ingestion_selection = config.choose_type_of_data_ingestion()
        print(data_ingestion_selection)
        if data_ingestion_selection == config.get_data_ingestion_config_library():
            print("Method returned")
        logger.info(f"Data ingestion type chosen: {config.get_data_ingestion_config_library()}")
        data_ingestion_config = config.get_data_ingestion_config_library()
        logger.info(f"Data ingestion type confirmed ")
        data_ingestion = DataIngestionLibrary(config=data_ingestion_config)
        logger.info(f"Data ingestion type configured in class DataIngestionLibrary:")
        data_ingestion.get_data_from_library()
        logger.info(f"Data ingestion performed")