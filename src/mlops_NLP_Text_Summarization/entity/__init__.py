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