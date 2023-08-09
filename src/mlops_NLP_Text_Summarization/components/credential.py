from mlops_NLP_Text_Summarization.constants import *
from mlops_NLP_Text_Summarization.utils.common import read_yaml, create_directories, save_json
from mlops_NLP_Text_Summarization.entity import CredentialsConfig



class Credentials:
    def __init__(
        self,
        secrets_filepath = SECRETS_FILE_PATH):


        self.secret= read_yaml(secrets_filepath)

    def get_mlflow_tracking_credentials(self) -> CredentialsConfig:
        secret = self.secret

        model_evaluation_config = CredentialsConfig(
            MLFLOW_TRACKING_URI=self.secret.MLFLOW_TRACKING_URI,
            MLFLOW_TRACKING_USERNAME=self.secret.MLFLOW_TRACKING_USERNAME,
            MLFLOW_TRACKING_PASSWORD = self.secret.MLFLOW_TRACKING_PASSWORD

        )
        return {
            "MLFLOW_TRACKING_URI": model_evaluation_config.MLFLOW_TRACKING_URI,
            "MLFLOW_TRACKING_USERNAME": model_evaluation_config.MLFLOW_TRACKING_USERNAME,
            "MLFLOW_TRACKING_PASSWORD": model_evaluation_config.MLFLOW_TRACKING_PASSWORD
        }