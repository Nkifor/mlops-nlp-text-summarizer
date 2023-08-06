from mlops_NLP_Text_Summarization.config.configuration import ConfigurationManager
from mlops_NLP_Text_Summarization.components.data_validation import DataValiadtion
from mlops_NLP_Text_Summarization.logging import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_files_exist()
