from mlops_NLP_Text_Summarization.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlops_NLP_Text_Summarization.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlops_NLP_Text_Summarization.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from mlops_NLP_Text_Summarization.pipeline.stage_04_model_training import ModelTrainingPipeline
from mlops_NLP_Text_Summarization.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from mlops_NLP_Text_Summarization.logging import logger


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> Stage | {STAGE_NAME} | Started")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage | {STAGE_NAME} | Finished \n\n ========================================")
except Exception as e:
    logger.exception(f">>>>> Stage | {STAGE_NAME} | Failed")
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>> Stage | {STAGE_NAME} | Started")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage | {STAGE_NAME} | Finished \n\n ========================================")
except Exception as e:
    logger.exception(f">>>>> Stage | {STAGE_NAME} | Failed")
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>> Stage | {STAGE_NAME} | Started")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage | {STAGE_NAME} | Finished \n\n ========================================")
except Exception as e:
    logger.exception(f">>>>> Stage | {STAGE_NAME} | Failed")
    raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>> Stage | {STAGE_NAME} | Started")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage | {STAGE_NAME} | Finished \n\n ========================================")
except Exception as e:
    logger.exception(f">>>>> Stage | {STAGE_NAME} | Failed")
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>> Stage | {STAGE_NAME} | Started")
    obj = ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage | {STAGE_NAME} | Finished \n\n ========================================")
except Exception as e:
    logger.exception(f">>>>> Stage | {STAGE_NAME} | Failed")
    raise e