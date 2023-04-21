ranking
==============================

The purpose of this project is to do a reproducible pipeline with a ranking similar questions based on [quora dataset](https://www.kaggle.com/c/quora-question-pairs). To make it reproducible and readable I want to use such technologies/libraries in my project:

- Reproducibility:
1. __Poetry__ — to keep track of dependencies and make it library <img height="18" width="28" src="https://img.shields.io/badge/-done-brightgreen" />
2. __Docker + FastAPI__ — to make a microservice with an isolated environment <img height="18" width="28" src="https://img.shields.io/badge/-todo-yellow" />
3. __mlflow__ - for keep track of experimentation <img height="18" width="28" src="https://img.shields.io/badge/-done-brightgreen" />

- Readability:
1. __Cookiecutter__’s DS project template — for easier navigation <img height="18" width="28" src="https://img.shields.io/badge/-done-brightgreen" />
2. I want to do __docstrings__ and explicitly write data types with __Typing__ <img height="18" width="28" src="https://img.shields.io/badge/-todo-yellow" />
3. Use __linters__ like __flake8__ or __black__ for codestyle <img height="18" width="28" src="https://img.shields.io/badge/-done-brightgreen" />

- CI/CD, tracking etc
1. __logging__ - for service tracking <img height="18" width="28" src="https://img.shields.io/badge/-todo-yellow" />
2. Do commits aligned with [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) <img height="18" width="28" src="https://img.shields.io/badge/-done-brightgreen" />


Stack
------------
<img height="32" width="32" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" /> <img height="32" width="64" src="https://miro.medium.com/max/633/0*oek9uPntF7vtHJP8.png" /> <img height="32" width="28" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/640px-PyTorch_logo_icon.svg.png" /> <img height="32" width="40" src="https://www.docker.com/wp-content/uploads/2022/03/vertical-logo-monochromatic.png" /> <img height="32" width="64" src="https://repository-images.githubusercontent.com/260928305/92388600-8d1c-11ea-9993-a726466b5099" /> <img height="32" width="48" src="https://engineering.fb.com/wp-content/uploads/2017/03/faiss_logo.png" />

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── .ignore files      <- To hide unnecessery data from push and build actions
    │
    └── poetry.lock        <- Poetry-related and env-related files
    └── pyproject.toml
    │
    └── Dockerfile         <- Dockerfile for building images

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
