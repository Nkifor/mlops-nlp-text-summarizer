import os
from mlops_NLP_Text_Summarization.logging import logger
from mlops_NLP_Text_Summarization.entity import DataValidationConfig


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config



    def validate_all_files_exist(self)-> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts","data_ingestion",f"{self.config.library_dataset_name}"))

            for file in all_files:
                if file not in self.config.all_required_files:
                    validation_status = False
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status})")
                else:
                    validation_status = True
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e