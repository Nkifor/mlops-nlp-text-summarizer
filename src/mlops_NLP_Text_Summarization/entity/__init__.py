from dataclasses import dataclass
from pathlib import Path

# below looks similatly to the config file data ingestion
@dataclass(frozen=True)
class DataIngestionConfigUnzipLink:
    root_dir: Path
    source_URL_zipped: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataIngestionConfigLink:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class DataIngestionConfigLibrary:
    root_dir: Path
    local_data_dir: Path
    library_dataset_name: str

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: str
    all_required_files_in_main_catalog: list
    library_dataset_name: str
    all_required_files_in_train_test_validations_catalogs: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class CredentialsConfig:
    MLFLOW_TRACKING_URI: str
    MLFLOW_TRACKING_USERNAME: str
    MLFLOW_TRACKING_PASSWORD: str



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    params: dict
    tokenizer_path: Path
    metric_file_name: Path
    #mlflow_uri: str
    experiment_name: str
    model_path_packed: Path