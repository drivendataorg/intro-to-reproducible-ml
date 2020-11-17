#!/usr/bin/env python
# coding: utf-8

# #  Introduction to Reproducible Machine Learning in Python
# 
# ## Tutorial Goals
# 
# This is a tutorial for Introduction to Reproducible Machine Learning in Python at [Good Tech Fest](https://www.goodtechfest.com/) Data Science Day November 2020. We will be working with dengue fever data from the [DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) practice machine learning competition on DrivenData.
# 
# This notebook is adapted from the [benchmark walkthrough blog post](https://www.drivendata.co/blog/dengue-benchmark/). This tutorial will walk you through the competition. We will show you how to load the data and do a quick exploratory analysis. Then, we will train a simple model, make some predictions, and then submit those predictions to the competition.
# 
# A secondary goal is to introduce tools and best practices for reproducibility along the way. These are our main tools, some of which we've already introduced:
# 
# 1. Reproducible environments with [conda](https://docs.conda.io/en/latest/), and environment manager.
# 2. A standardized, flexible project structure with [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).
# 2. Easy code review and change tracking with [nbautoexport](https://github.com/drivendataorg/nbautoexport).
# 3. [Jupyter](https://jupyter.org/) notebooks/lab for [literate programming](https://en.wikipedia.org/wiki/Literate_programming).
# 
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

PROJ_ROOT = Path().resolve().parent
DATA_DIR = PROJ_ROOT / "data" / "raw"
print("PROJ_ROOT :", PROJ_ROOT)
print("DATA_DIR :", DATA_DIR)


# As part of the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure, we recommend keeping data in a `data/` directory, and further having a `data/raw/` subdirectory where the raw data lives unchanging.
# 
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

print('\nIquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)


# The [problem description](https://www.drivendata.org/competitions/44/page/82/) gives a good overview of the available variables, but we'll look at the head and types of the data here as well:

sj_train_features.head()


sj_train_features.dtypes


# There are _a lot_ of climate variables here, but the first thing that we'll note is that the `week_start_date` is included in the feature set. This makes it easier for competitors to create time based features, but for this first-pass model, we'll drop that column since we shouldn't use it as a feature in our model.

# Remove `week_start_date` string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)


# Next, let's check to see if we are missing any values in this dataset:

# Null check
pd.isnull(sj_train_features).any()


# Let's plot one of our features to get a sense of what it looks like.

(sj_train_features
     .ndvi_ne
     .plot
     .line(lw=0.8))

plt.title('Vegetation Index over Time')
plt.xlabel('Time')


# Since these are time-series, we can see the gaps where there are `NaN`s by plotting the data. Since we can't build a model without those values, we'll take a simple approach and just fill those values with the most recent value that we saw up to that point. This is probably a good part of the problem to improve your score by getting smarter.

# impute data using the forward fill method (propagate last valid observation forward to next valid)
sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)

pd.isnull(sj_train_features).any()


# ## Distribution of labels
# 
# Our target variable, `total_cases` is a non-negative integer, which means we're looking to make some **count predictions**. Standard regression techniques for this type of prediction include
# 
# 1. Poisson regression
# 2. Negative binomial regression
# 
# Which techniqe will perform better depends on many things, but the choice between Poisson regression and negative binomial regression is pretty straightforward. Poisson regression fits according to the assumption that the mean and variance of the population distributiona are equal. When they aren't, specifically when the variance is much larger than the mean, the negative binomial approach is better. Why? It isn't magic. The negative binomial regression simply lifts the assumption that the population mean and variance are equal, allowing for a larger class of possible models. In fact, from this perspective, the Poisson distribution is but a special case of the negative binomial distribution.
# 
# Let's see how our labels are distributed!

print('San Juan')
print('mean: ', sj_train_labels.mean()[0])
print('var :', sj_train_labels.var()[0])

print('\nIquitos')
print('mean: ', iq_train_labels.mean()[0])
print('var :', iq_train_labels.var()[0])


# It's looking like a negative-binomial sort of day in these parts.

sj_train_labels.hist()


iq_train_labels.hist()


# ### `variance  >>  mean` suggests `total_cases` can be described by a negative binomial distribution, so we'll use a negative binomial regression below.

# ## Which inputs strongly correlate with `total_cases`?
# 
# Our next step in this process will be to select a subset of features to include in our regression. Our primary purpose here is to get a better understanding of the problem domain rather than eke out the last possible bit of predictive accuracy. The first thing we will do is to add the `total_cases` to our dataframe, and then look at the correlation of that variable with the climate variables.

sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases


# Compute the data correlation matrix.

# compute the correlations
sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()


# plot san juan
sj_corr_heat = sns.heatmap(sj_correlations)
plt.title('San Juan Variable Correlations')


# plot iquitos
iq_corr_heat = sns.heatmap(iq_correlations)
plt.title('Iquitos Variable Correlations')


# ### Many of the temperature data are strongly correlated, which is expected. But the `total_cases` variable doesn't have many obvious strong correlations.
# 
# Interestingly, `total_cases` seems to only have weak correlations with other variables. Many of the climate variables are much more strongly correlated. Interestingly, the vegetation index also only has weak correlation with other variables. These correlations may give us some hints as to how to improve our model that we'll talk about later in this post. For now, let's take a `sorted` look at `total_cases` correlations.

# San Juan
(sj_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())


# Iquitos
(iq_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())


# ### A few observations
# 
# #### The wetter the better
# * The correlation strengths differ for each city, but it looks like `reanalysis_specific_humidity_g_per_kg` and `reanalysis_dew_point_temp_k` are the most strongly correlated with `total_cases`. This makes sense: we know mosquitos thrive _wet_ climates, the wetter the better!
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
# Nevertheless, for this benchmark we'll focus on the linear __wetness__ trend we see above, and reduce our inputs to...
# 
# #### a few good variables:
# 
# * `reanalysis_specific_humidity_g_per_kg`
# * `reanalysis_dew_point_temp_k`
# * `station_avg_temp_c`
# * `station_min_temp_c`

# ## A mosquito model
# 
# Now that we've explored this data, it's time to start modeling. Our first step will be to build a function that does all of the preprocessing we've done above from start to finish. This will make our lives easier, since it needs to be applied to the test set and the traning set before we make our predictions.
