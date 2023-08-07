# Deployment NLP model for Text Summarization



## Description and shortened context of the training data


## Stages of the project

1. Planning desired outcome
2. Configuration of environment setup
3. Preparation of notebook every step and refactoring to modular form
4. Initialization DVC repository
5. Prediction expreriments
6. Deployment of model


## Iterative workflows for particular parts of the project

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml




## How to run

### STEPS:

Clone the repository

```bash
https://github.com/Nkifor/mlops-nlp-text-summarizer
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n venvml python=3.8 -y
```

```bash
conda activate venvml
```

### STEP 02 - install the requirements
```bash
pip install -r requirements.txt
```

