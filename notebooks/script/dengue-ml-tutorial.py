#!/usr/bin/env python
# coding: utf-8

# #  Introduction to Reproducible Machine Learning in Python
# 
# ## Tutorial Goals
# 
# This is a tutorial for Introduction to Reproducible Machine Learning in Python at [Good Tech Fest](https://www.goodtechfest.com/) Data Science Day November 2020. We will be working with dengue fever data from the [DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) practice machine learning competition on DrivenData.
# 
# This notebook is adapted from the [benchmark walkthrough blog post](https://www.drivendata.co/blog/dengue-benchmark/). This tutorial will walk you through the competition. We will show you how to load the data and do a quick exploratory analysis. Then, we will train a simple model, make some predictions, and finally submit those predictions to the competition.
# 
# A secondary goal of this tutorial is to introduce tools and best practices for reproducibility along the way. These are our main tools, which allow you to reproduce...
# 1. **environments:** Using an environment manager like [conda](https://docs.conda.io/en/latest/) allows you to avoid running into pesky dependency conflicts down the road.
# 2. **project structure**: [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) allows for a standardized, flexible project structure. That means you can find the code you're looking for more quickly and easily in any project.
# 3. **the past**: Notebooks are great for analysis and exploration, but less so for code reviews and tracking changes. [nbautoexport](https://github.com/drivendataorg/nbautoexport) simplifies that process by automatically exporting notebooks to more easily diffable scripts.
# 4. **analysis**: [Jupyter](https://jupyter.org/) notebooks/lab are a great tool for [literate programming](https://en.wikipedia.org/wiki/Literate_programming). Individually executable cells allow you to weave narrative, code, and visualizations together.
# 5. **ML pipelines**: We recommend using [scikit-learn](https://scikit-learn.org/) for self-contained, reusable ML pipelines.
# 
# ## Problem Introduction
# ----------------------
# 
# Dengue fever is bad. It's real bad. Dengue is a mosquito-borne disease that occurs in tropical and sub-tropical parts of the world. In mild cases, symptoms are similar to the flu: fever, rash and muscle and joint pain. But severe cases are dangerous, and dengue fever can cause severe bleeding, low blood pressure and even death.
# 
# Because it is carried by mosquitoes, the transmission dynamics of dengue are related to climate variables such as temperature and precipitation. Although the relationship to climate is complex, a growing number of scientists argue that climate change is likely to produce distributional shifts that will have significant public health implications worldwide.
# 
# We've [launched a competition](https://www.drivendata.org/competitions/44/) to use open data to predict the occurrence of Dengue based on climatological data. Here's a look at the data and how to get started!

# ### Import our tools and data
# 
# As always, we begin with the sacred `import`'s of data science:

from pathlib import Path

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# turn off warnings just for the sake of this tutorial
from warnings import filterwarnings
filterwarnings('ignore')


# ## A Tale of Two Cities
# 
# ![](https://community.drivendata.org/uploads/default/original/1X/4c3a8204d1715b5e2ee24da78abbad1515eccd5f.png)
# 
# This dataset has two cities in it: San Juan, Puerto Rico (right) and Iquitos, Peru (left). Since we hypothesize that the spread of dengue may follow different patterns between the two, we will divide the dataset, train separate models for each city, and then join our predictions before making our final submission.

# As part of the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure, we recommend keeping data in a `data/` directory, and further having a `data/raw/` subdirectory where the raw data lives unchanging.

PROJ_ROOT = Path().resolve().parent
DATA_DIR = PROJ_ROOT / "data" / "raw"
print("PROJ_ROOT :", PROJ_ROOT)
print("DATA_DIR :", DATA_DIR)


# Let's load the data from that path and take a look at our features.

# load the provided data

train_features = pd.read_csv(DATA_DIR / 'dengue_features_train.csv',
                             index_col=[0,1,2])

train_labels = pd.read_csv(DATA_DIR / 'dengue_labels_train.csv',
                           index_col=[0,1,2])


train_features


train_labels


# Since we suspect that the dengue fever patterns will be different for each city, let's separate the data using the `city` index.

# Separate data for San Juan
sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

# Separate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']


# ### Data Exploration
# 
# Now that we've loaded up our data, we can start to explore what it looks like.

print('San Juan')
print('features: ', sj_train_features.shape)
print('labels  : ', sj_train_labels.shape)
print()
print('Iquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)


# The [problem description](https://www.drivendata.org/competitions/44/page/82/) gives a good overview of the available variables, but we'll look at the head and types of the data here as well:

sj_train_features.head()


sj_train_features.dtypes


# #### Remove `week_start_date`

# There are _a lot_ of climate variables here, but the first thing that we'll note is that the `week_start_date` is included in the feature set. This makes it easier for competitors to create time based features, but for this first-pass model, we'll drop that column since we shouldn't use it as a feature in our model.

# Remove `week_start_date` string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)


# #### Check for Missing Values

# Next, let's check to see if we are missing any values in this dataset:

# Null check
pd.isnull(sj_train_features).any()


# Let's plot one of our features to get a sense of what it looks like.

(sj_train_features
     .ndvi_ne
     .plot
     .line(lw=0.8))

plt.title('Vegetation Index over Time')
plt.xlabel('Time');


# #### Impute Data

# Since these are time-series, we can see the gaps where there are `NaN`s by plotting the data. Since we can't build a model without those values, we'll take a simple approach and just fill those values with the most recent value that we saw up to that point. This is probably a good part of the problem to improve your score by getting smarter.

# impute data using the forward fill method (propagate last valid observation forward to next valid)
sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)

pd.isnull(sj_train_features).any()


# ### Distribution of Labels
# 
# Our target variable, `total_cases` is a non-negative integer, which means we're looking to make some **count predictions**. Let's see how our labels are distributed!

sj_train_labels.hist()
plt.title('Total Cases in San Juan');


iq_train_labels.hist()
plt.title('Total Cases in Iquitos');


# These sorts of right-skewed distributions are pretty typical for count data.

# ### Which inputs strongly correlate with `total_cases`?
# 
# Our next step in this process will be to select a subset of features to include in our regression. Our primary purpose here is to get a better understanding of the problem domain rather than eke out the last possible bit of predictive accuracy.
# 
# The first thing we will do is to add the `total_cases` to our dataframe, and then look at the correlation of that variable with the climate variables.

sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases


# #### Compute the data correlation matrix

# compute the correlations
sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()


# plot san juan
fig, ax = plt.subplots(figsize=(10,8))
sj_corr_heat = sns.heatmap(sj_correlations, ax=ax)
plt.xticks(rotation=45, ha='right') 
plt.title('San Juan Variable Correlations');


# plot san juan
fig, ax = plt.subplots(figsize=(10, 8))
iq_corr_heat = sns.heatmap(iq_correlations, ax=ax)
plt.xticks(rotation=45, ha='right') 
plt.title('Iquitos Variable Correlations');


# ### Many of the temperature data are strongly correlated, which is expected. But the `total_cases` variable doesn't have many obvious strong correlations.
# 
# Interestingly, `total_cases` seems to only have weak correlations with other variables. Many of the climate variables are much more strongly correlated. The vegetation index (`ndvi`) also only has weak correlation with other variables. These correlations may give us some hints as to how to improve our model that we'll talk about later in this post. For now, let's take a `sorted` look at `total_cases` correlations.

# San Juan
(sj_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh());


# Iquitos
(iq_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh());


# ### A few observations
# 
# #### The wetter the better
# * The correlation strengths differ for each city, but it looks like `reanalysis_specific_humidity_g_per_kg` and `reanalysis_dew_point_temp_k` are the most strongly correlated with `total_cases`. This makes sense: we know mosquitos thrive in _wet_ climates, the wetter the better!
# 
# #### Hot and heavy
# * As we all know, "cold and humid" is not a thing. So it's not surprising that as minimum temperatures, maximum temperatures, and average temperatures rise, the `total_cases` of dengue fever tend to rise as well.
# 
# #### Sometimes it rains, so what
# * Interestingly, the `precipitation` measurements bear little to no correlation to `total_cases`, despite strong correlations to the `humidity` measurements, as evident by the heatmaps above.
# 
# ### This is just a first pass
# 
# Precisely _none_ of these correlations are very strong. Of course, that doesn't mean that some **feature engineering wizardry** can't put us in a better place **(`standing_water` estimate, anyone?)**. Also, it's always useful to keep in mind that **life isn't linear**, but out-of-the-box correlation measurement is – or at least, it measures linear dependence.
# 
# Nevertheless, for this tutorial we'll focus on the linear __wetness__ trend we see above, and reduce our inputs to...
# 
# #### a few good variables:
# 
# * `reanalysis_specific_humidity_g_per_kg`
# * `reanalysis_dew_point_temp_k`
# * `station_avg_temp_c`
# * `station_min_temp_c`

# ## Let's (scikit)-learn
# 
# Now that we've explored this data and picked some input variables, it's time to start modeling. 
# 
# We will be using the [**scikit-learn**](https://scikit-learn.org/stable/) (a.k.a. sklearn) library for machine learning. It's a mature library with a comprehensive set of tools and good documentation. It also has an opinionated design that emphasizes reuse and reproducibility through its object-oriented style and "pipeline" concepts. We highly recommend starting with scikit-learn for most machine learning situations.

# ### Choosing a modeling approach

# When doing a first-pass model, it's usually a good idea to start with a linear model. As discussed earlier, we're trying to predict **counts**. The generalized linear model family has two models for modeling counts:
# 
# 1. Poisson regression
# 2. Negative binomial regression
# 
# Poisson regression is the simpler model and fits according to the assumption that the mean and variance of the population distribution are equal. 
# 
# When the mean and variance aren't equal, specifically when the variance is much larger than the mean, the negative binomial approach is better. Why? It isn't magic. The negative binomial regression simply lifts the assumption that the population mean and variance are equal, allowing for a larger class of possible models. In fact, from this perspective, the Poisson distribution is but a special case of the negative binomial distribution.
# 
# Since scikit-learn has a [`PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html) implementation, but not one for negative binomial regression, we'll be doing Poisson regression for this first-pass model.
# 
# We can, however, check the assumptions really quickly:

print('San Juan')
print('mean: ', sj_train_labels.mean()[0])
print('var :', sj_train_labels.var()[0])

print('\nIquitos')
print('mean: ', iq_train_labels.mean()[0])
print('var :', iq_train_labels.var()[0])


# It does indeed look like the variance is much higher than the mean for both city's count distributions! This suggests that negative binomial distribution is a better choice statistically, and we have an obvious avenue for the next iteration.

# Let's get started on the modeling by importing the parts of scikit-learn that we will be using.

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.model_selection import train_test_split

from sklearn.linear_model import PoissonRegressor

# For pipeline visualization
from sklearn import set_config
set_config(display='diagram') 


# ### Loading and splitting data

# First, let's load the training data again so we have clean, fresh dataframes that haven't been modified. 

# Let's make a function since we'll be splitting by city a lot
def load_data(data_path):
    """Load data, sorts by time, and split by city"""
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    return df.loc["sj"], df.loc["iq"]

sj_features_df, iq_features_df = load_data(DATA_DIR / "dengue_features_train.csv")
sj_labels_df, iq_labels_df = load_data(DATA_DIR / "dengue_labels_train.csv")

print("sj_features_df", sj_features_df.shape)
print("iq_features_df", iq_features_df.shape)
print("sj_labels_df", sj_labels_df.shape)
print("iq_labels_df", iq_labels_df.shape)


# Next, we'll split our data into training and evaluation. It's important when evaluating model performance to do so on data that was not used for training. Evaluating and training on the same data leads to an overconfident evaluation. Since we have time series data, we'll do a strict-future holdout for our evaluation. Our data is already sorted by time, so we can use sklearn's `train_test_split` function _without shuffling_.

sj_X_train, sj_X_eval, sj_y_train, sj_y_eval = train_test_split(
    sj_features_df,
    sj_labels_df,
    test_size=0.25,
    shuffle=False,
)

iq_X_train, iq_X_eval, iq_y_train, iq_y_eval = train_test_split(
    iq_features_df,
    iq_labels_df,
    test_size=0.25,
    shuffle=False,
)


# ### Data Preprocessing

# Now let's define our preprocessing steps. We'll be doing two main things:
# - **Scaling**: This step is not critical but often useful to generally do as a rule of thumb. In our case, scaling will help the model regularize features more effectively. It will also give us model weights that can be compared across features and interpreted as feature importances. 
# - **Missing Value Imputation**: This step is necessary. Most machine learning models don't handle missing values. Because we have a time series problem, we'll use forward-fill imputation as a first pass approach, where we fill using the previous timestep's value. 

# sklearn doesn't have a forward-fill imputation transformer, so we'll need to make our own. This is a good demonstration of sklearn's `FunctionTransformer` class, which you can use to implement simple functions into sklearn transformers. We define a function `forward_fill` which takes a numpy array, uses the `ffill` functionality from pandas, and then returns the data back as a numpy array.

def forward_fill(array):
    return pd.DataFrame(array).ffill().values

forward_imputer = FunctionTransformer(forward_fill)


# Next, we'll stitch together our preprocessing steps into a preprocessor object that can be fitted and reused. We will use the `ColumnTransformer` class, which is the way sklearn expects you to pull columns out of a pandas dataframe and apply data transformations. In setting up the `ColumnTransformer`, we chain together our two preprocessing steps into one transformer using `Pipeline`, and then we set that we want to apply it to our subset of features. We will have the `ColumnTransformer` drop the remainder of the dataframe columns. 

features_to_use = [
    "reanalysis_specific_humidity_g_per_kg", 
    "reanalysis_dew_point_temp_k", 
    "station_avg_temp_c", 
    "station_min_temp_c"
]

preprocessing_steps = Pipeline([
    ("standard_scaler", StandardScaler()),
    ("forward_imputer", forward_imputer)
])

sj_preprocessor = ColumnTransformer(
    transformers = [
        ("features", preprocessing_steps, features_to_use)
    ],
    remainder = "drop"
)


# We actually will need a second preprocessor object. This is because the scaling is fit to data, so we need one for San Juan and another for Iquitos. You'll notice we named the preprocessor we just created `sj_preprocessor`. We can use the `clone` function make a copy for `iq`. 

iq_preprocessor = clone(sj_preprocessor)


# ## Pipeline

# Now we'll wrap up our full modeling workflow, including the Poisson Regression model itself, into sklearn `Pipeline` objects. We used these earlier to chain our scaling and imputation. Pipelines encapsulate everything into one object with one interface, which makes it easy to train a model, save it to one file for later (e.g., by [pickling](https://docs.python.org/3/library/pickle.html)), reload it, and use it again. It's a valuable tool in making your machine learning models reusable and reproducible.

sj_pipeline = Pipeline([
    ("preprocessor", sj_preprocessor),
    ("estimator", PoissonRegressor())
])

iq_pipeline = Pipeline([
    ("preprocessor", iq_preprocessor),
    ("estimator", PoissonRegressor())
])

iq_pipeline


# Now, let's fit our models and evaluate them!

sj_pipeline.fit(sj_X_train, sj_y_train)
iq_pipeline.fit(iq_X_train, iq_y_train)

sj_preds = sj_pipeline.predict(sj_X_eval)
iq_preds = iq_pipeline.predict(iq_X_eval)


# We'll plot the predictions against the true case counts for both models. 

figs, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,7))

sj_y_eval.plot(ax=axes[0],legend=None)
axes[0].plot(sj_preds, label="Predictions")
axes[0].set_ylabel("SJ Case Count")

iq_y_eval.rename(columns={"total_cases": "Actual"}).plot(ax=axes[1])
axes[1].plot(iq_preds, label="Predictions")
axes[1].set_ylabel("IQ Case Count")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
None


# ## Reflecting on our performance
# 
# These graphs can actually tell us a lot about where our model is going wrong and give us some good hints about where investments will improve the model performance. 
# 
# For example, we see that our model in orange does track the seasonality of dengue cases. However, the timing of the seasonality of our predictions has a mismatch with the actual results. One potential reason for this is that our features don't look far enough into the past—that is to say, we are asking to predict cases at the same time as we are measuring percipitation. Because dengue is mosquito-borne, and the mosquito lifecycle depends on water, we need to take both the life of a mosquito and the time between infection and symptoms into account when modeling dengue. This is a critical avenue to explore when improving this model.
# 
# The other important error is that our predictions are relatively consistent—we miss the spikes that are large outbreaks. One reason is that we don't take into account the contagiousness of dengue. A possible way to account for this is to build a model that progressively predicts a new value while taking into account the previous prediction. By training on the dengue outbreaks and then using the predicted number of patients in the week before, we can start to model this time dependence that the current model misses.
# 
# We also know that Poisson regression is not the best-suited for this dataset. We saw earlier than our variance in counts is higher than the mean, making it negative binomial regression more appropriate. While sklearn doesn't offer negative binomial regression, we could use the [statsmodel](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.NegativeBinomial.html) package. We can also take advantage of sklearn's extendability to wrap statsmodel's `NegativeBinomial` model inside custom sklearn estimator, so that we can plug it into our sklearn pipeline. More about custom sklearn estimators is available in their [documentation](https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator). 
# 
# So, we know we're not going to win this thing, but let's submit the predictions anyway!

# ### Generating a submission

# First, we want to retrain our models on the full dataset.

sj_pipeline.fit(sj_features_df, sj_labels_df)
iq_pipeline.fit(iq_features_df, iq_labels_df)
None # Don't print diagram


# Now, we'll load our test data and generate predictions.

sj_test_df, iq_test_df = load_data(DATA_DIR / "dengue_features_test.csv")

print("sj_test_df", sj_test_df.shape)
print("iq_test_df", iq_test_df.shape)


sj_test_preds = sj_pipeline.predict(sj_test_df)
iq_test_preds = iq_pipeline.predict(iq_test_df)


# Then, we'll load up the submission format and plug our predictions in. 

OUTPUT_DIR = PROJ_ROOT / "data" / "processed"
print("OUTPUT_DIR : ", OUTPUT_DIR)


sj_submission_df, iq_submission_df = load_data(DATA_DIR / "submission_format.csv")

sj_submission_df["total_cases"] = sj_test_preds.astype(int)
iq_submission_df["total_cases"] = iq_test_preds.astype(int)

# Rejoin SJ and IQ together. 
submission_df = pd.concat({
    "sj": sj_submission_df, 
    "iq": iq_submission_df
}, names=["city"])

# order reorder based on original order

submission_df.to_csv(OUTPUT_DIR / "submission.csv")

submission_df.head(10)


# Head to the [competition submission page](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/submissions/) and submit the predictions.

# <img src="https://drivendata-public-assets.s3.amazonaws.com/gtf-intro-to-reproducible-ml-submission.png" alt="Submission" width="500" />

# Alright, it's a start! 
