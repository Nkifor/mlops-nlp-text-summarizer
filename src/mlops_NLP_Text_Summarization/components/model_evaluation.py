from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, load_metric
import torch
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
import json
import joblib
import yaml
import os
import logging
from pathlib import Path
import mlflow
from mlops_NLP_Text_Summarization.entity import ModelEvaluationConfig
from mlops_NLP_Text_Summarization.constants import *
from mlops_NLP_Text_Summarization.utils.common import save_json




class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config



    def generate_batch_sized_chunks(self,list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]


    def calculate_metric_on_test_ds(self,dataset, metric, model, tokenizer,
                               batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu",
                               column_text="article",
                               column_summary="highlights"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):

            inputs = tokenizer(article_batch, max_length=1024,  truncation=True,
                            padding="max_length", return_tensors="pt")

            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device),
                            length_penalty=0.8, num_beams=8, max_length=128)
            ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''

            # Finally, we decode the generated texts,
            # replace the  token, and add the decoded texts with the references to the metric.
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
                for s in summaries]

            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]


            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        #  Finally compute and return the ROUGE scores.
        score = metric.compute()
        return score


    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        #loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)


        #name of experiment
        if mlflow.active_run():
            mlflow.end_run()


        mlflow.set_experiment(self.config.experiment_name)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        def get_experiment_id(name):
            exp = mlflow.get_experiment_by_name(name)
            if exp is None:
                exp_id = mlflow.create_experiment(name)
                return exp_id
            return exp.experiment_id


        def get_last_run_id(exp_name):
            exp = get_experiment_id(exp_name)
            client = mlflow.MlflowClient()
            runs = client.search_runs(experiment_ids=exp)
            if len(runs) == 0:
                return None
            last_run = runs[0]
            return last_run.info.run_id

            run_id = get_last_run_id(self.config.experiment_name)
            print(run_id)
            os.environ[sys.argv[1]] = run_id
            print(f'Saving {run_id} to environment variable {sys.argv[1]}')

            run_id = os.environ[sys.argv[1]]


# Select a name for the model to be registered



        with mlflow.start_run(run_name=self.config.experiment_name):

            run_id = get_last_run_id(self.config.experiment_name)

            subpath = "text_summarization"

            model_name = "samsum_pegasus_model"

            # build the run URI
            run_uri = f'runs:/{run_id}/{subpath}'

            # register the model



            rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

            rouge_metric = load_metric('rouge')

            score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10], rouge_metric, model_pegasus, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary= 'summary'
                )

            rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )


            print(rouge_dict)
            #df = pd.DataFrame(rouge_dict, index = ['pegasus'] )


            save_json(path=Path(self.config.metric_file_name), data=rouge_dict)
            mlflow.log_params(self.config.params)
            mlflow.log_metric('rouge1', rouge_dict['rouge1'])
            mlflow.log_metric('rouge2', rouge_dict['rouge2'])
            mlflow.log_metric('rougeL', rouge_dict['rougeL'])
            mlflow.log_metric('rougeLsum', rouge_dict['rougeLsum'])

            model_version = mlflow.register_model(run_uri, model_name)


        if tracking_url_type_store != "file":
            mlflow.pytorch.log_model(model_pegasus, artifact_path= self.config.model_path, registered_model_name="samsum_pegasus_model_reg_name")
        else:
            print( "logging model to local file")