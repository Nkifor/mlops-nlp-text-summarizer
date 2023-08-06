import os
from mlops_NLP_Text_Summarization.logging import logger
from mlops_NLP_Text_Summarization.entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self)-> bool:
        try:
            validation_status = None

            all_files_in_main_catalog = os.listdir(os.path.join("artifacts","data_ingestion",f"{self.config.library_dataset_name}"))
            files_in_train_test_validations_catalogs = os.listdir(os.path.join("artifacts","data_ingestion",f"{self.config.library_dataset_name}","train"))
            files_in_train_test_validations_catalogs.extend(os.listdir(os.path.join("artifacts","data_ingestion",f"{self.config.library_dataset_name}","test")))
            files_in_train_test_validations_catalogs.extend(os.listdir(os.path.join("artifacts","data_ingestion",f"{self.config.library_dataset_name}","validation")))

            for file in all_files_in_main_catalog:
                if file not in self.config.all_required_files_in_main_catalog:
                    validation_status = False
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status})")
                    for file in self.config.all_required_files_in_train_test_validations_catalogs:
                        if file not in files_in_train_test_validations_catalogs:
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