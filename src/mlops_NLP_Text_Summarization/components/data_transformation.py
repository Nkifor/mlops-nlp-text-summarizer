import os
from mlops_NLP_Text_Summarization.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict
from mlops_NLP_Text_Summarization.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self,example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length = 128, truncation = True )

        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        data_path = self.config.data_path
        train_dataset = load_from_disk(data_path + "/train")
        test_dataset = load_from_disk(data_path + "/test")
        validation_dataset = load_from_disk(data_path + "/validation")
        dataset_samsum = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum"))