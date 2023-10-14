# Deployment NLP model for Text Summarization

## Description and shortened context of the project
Project assumes to perform text summarizing task by apply NLP model. The model is based on HuggingFace transformers library and is fine tuned on CNN/Daily Mail dataset. The model is deployed via FastAPI and Dockerized. The model is deployed on Heroku. The project is modularized and can be easily extended. The project is ready to be used as a template for other NLP tasks.

## Project structure - main processes and their description


- Data Ingestion - data is gathered from a huggingface dataset and is split into train and test
- Data Validation - data is validated to check if it is in the correct format
- Data Transformation - data is transformed into a format that can be used by the model
- Model Training - model is trained on the transformed data
- Model Evaluation - model is evaluated on the test data
- Model Fine Tuning - model is fine tuned on the test data
- Model Deployment via FastAPI
- Dockerization of the model
- Running the model via Heroku



## Technologies used:

![HuggingFace](https://img.shields.io/badge/huggingface-%23F37626.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-%2300C7B7.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)




## Stages of the project

1. Planning desired outcome
2. Configuration of environment setup
3. Preparation of notebook every step and refactoring to modular form
4. Prediction expreriments
5. Model fine tuning
6. Dockerization of the local model via FastAPI
7. Deployment of the model on Heroku



## Iterative workflows for particular parts of the project

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py





## How to reproduce the project

### STEPS:

Clone the repository

```bash
https://github.com/Nkifor/mlops-nlp-text-summarizer
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n venvml python=3.8 -y
```



### STEP 02 - install the requirements
```bash
pip install -r requirements.txt
```


### STEP 03 - update setup and logic and create your own setup file

### STEP 04 - Run a pipelines - ingestion, validation, training, evaluation

```bash
python main.py
```

### STEP 05 - After evaluation run the model fine tuning in the notebook to get the best model

NLP_Summarization_Pegasus_model.ipynb


### STEP 06 - Dockerization of the local model via FastAPI

First command is to build the image and the second one is to run the image and check if model works in the browser as local environment

```bash
docker build -t nkfortextsummarizer .

docker run -p 80:7000 nkfortextsummarizer
```




