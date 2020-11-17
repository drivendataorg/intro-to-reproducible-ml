intro-to-reproducible-ml
==============================

Materials for tutorial @ Good Tech Fest DS - Nov. 2020

This repository contains materials for the "Introduction to Reproducible Machine Learning in Python" tutorial session at [Good Tech Fest Data Science Day 2020](https://www.goodtechfest.com/home).

> Come ready to get your data science on. The goal of this workshop is to build your first machine learning model while learning the best tools and practices for reproducibility. You’ll need a laptop and some enthusiasm to get started. We will start from raw data and end by making a set of predictions. This will be an applied workshop looking at a real problem. We’ll be modeling the spread of dengue fever based on time, climate, and location. This will be based on a competition running on DrivenData. The primary goal is for participants to understand that machine learning isn’t magic. In the time it takes to run a tutorial, we can load a dataset, train a model, and make predictions. The secondary goal is to introduce participants to best practices for reproducible machine learning throughout the modeling pipeline. One of the focuses of the example will be demonstrating the resources where participants can learn more about each of the steps in the process.

If you are participating through the conference, please join the `#introduction-to-machine-learning-in-python` Slack channel.

Note that in order to get the data, you should [sign up for the DengAI: Predicting Disease Spread competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) on drivendata.org.

## Setting up

### Setting up this project

To get this repository, the best way is to have `git` and use `git clone`:

```bash
git clone https://github.com/drivendataorg/intro-to-reproducible-ml.git
```

then to enter the project
```bash
cd intro-to-reproducible-ml
```

This project is a simplified version of what is generated using [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/), a standardized structure we recommend for data science projects.

### Getting the data

To access the data, please sign up for the [DengAI: Predicting Disease Spread competition on drivendata.org](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/). Then, you can find the data on the [data download page](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/data/).

### Setting up the environment

This project requires Python 3.

Virtual environments are important for creating reproducible analyses. One popular tool for managing Python and virtual environments is [`conda`](https://docs.conda.io/en/latest/miniconda.html). You can set up the environment for this project with `conda` using the commands below.

```bash
conda create -n intro-to-reproducible-ml python=3.7
conda activate intro-to-reproducible-ml
pip install -r requirements.txt
```

### Running the notebook

The analysis is saved as a Jupyter notebook file. You can use [Jupyter Lab](https://jupyter.org/) to view and edit it:

```bash
jupyter lab
```



Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
