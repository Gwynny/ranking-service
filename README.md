ranking
==============================

The purpose of this project is to do a reproducible pipeline with a ranking similar questions based on [quora dataset](https://www.kaggle.com/c/quora-question-pairs). To make it reproducible and readable I want to use such technologies/libraries in my project:

1. __Poetry__ — to keep track of dependencies
2. __Docker + FastAPI__ — to make a microservice with an isolated environment
3. __Cookiecutter__’s DS project template — for easier navigation
4. __logging__ - for service tracking
5. I want to do __docstrings__ and explicitly write data types of all incoming parameters and outputs of functions
6. Make it a library with setuptools or Poetry
7. Use __linters__ for codestyle
8. Do commits aligned with [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)

Optional (if I will have time):
1. __mlflow__ - for keep track of experimentation

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
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
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
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
