from mlops_NLP_Text_Summarization.config.configuration import ConfigurationManager
from mlops_NLP_Text_Summarization.components.model_training import ModelTraining
from mlops_NLP_Text_Summarization.logging import logger


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_trainer_config = ModelTraining(config=model_training_config)
        model_trainer_config.train()
