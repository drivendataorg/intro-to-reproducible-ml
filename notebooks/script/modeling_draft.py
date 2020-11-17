#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJ_ROOT = Path().resolve().parent
RAW_DATA_DIR = PROJ_ROOT / "data" / "raw"
print(PROJ_ROOT)
print(RAW_DATA_DIR)


def load_data(data_path):
    """Load data, sorts by time, and split by city"""
    df = pd.read_csv(data_path, index_col=[0, 1, 2]).sort_index()    
    return df.loc["sj"], df.loc["iq"]


# ## Training some models
# 
# Now that we've explored this data, it's time to start modeling. We will be using the scikit-learn (a.k.a. sklearn) library for machine learning. It's a mature library with a comprehensive set of tools, a thoughtful opinionated design, and good documentation. We highly recommend starting with scikit-learn for machine learning.
# 
# Let's import the parts of scikit-learn that we will be using.

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.model_selection import train_test_split

from sklearn.linear_model import PoissonRegressor


# Let's load the training data again so we have clean, fresh dataframes that haven't been modified. 

sj_features_df, iq_features_df = load_data(RAW_DATA_DIR / "dengue_features_train.csv")
sj_labels_df, iq_labels_df = load_data(RAW_DATA_DIR / "dengue_labels_train.csv")

print("sj_features_df", sj_features_df.shape)
print("iq_features_df", iq_features_df.shape)
print("sj_labels_df", sj_labels_df.shape)
print("iq_labels_df", iq_labels_df.shape)


# Now let's define our preprocessing steps. We'll be doing two 

def forward_fill(array):
    return pd.DataFrame(array).ffill().values

forward_imputer = FunctionTransformer(forward_fill)

preprocessing_steps = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('forward_imputer', forward_imputer)
])


features_to_use = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c']

sj_preprocessor = ColumnTransformer(
    transformers = [
        ("features", preprocessing_steps, features_to_use)
    ],
    remainder = "drop"
)

iq_preprocessor = clone(sj_preprocessor)


# Now we can take a look at the smaller dataset and see that it's ready to start modelling:

# ## Split it up!

# Since this is a timeseries model, we'll use a strict-future holdout set when we are splitting our train set and our test set. We'll keep around three quarters of the original data for training and use the rest to test. We'll do this separately for our San Juan model and for our Iquitos model.

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


# ## Training time
# 
# This is where we start getting down to business. As we noted above, we'll train a NegativeBinomial model, which is often used for count data where the mean and the variance are very different. In this function we have three steps. The first is to specify the functional form 

sj_pipeline = Pipeline([
    ("preprocessor", sj_preprocessor),
    ("estimator", PoissonRegressor())
])

iq_pipeline = Pipeline([
    ("preprocessor", iq_preprocessor),
    ("estimator", PoissonRegressor())
])


sj_pipeline.fit(sj_X_train, sj_y_train)
iq_pipeline.fit(iq_X_train, iq_y_train)

sj_preds = sj_pipeline.predict(sj_X_eval)
iq_preds = iq_pipeline.predict(iq_X_eval)


sj_y_eval


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
# These graphs can actually tell us a lot about where our model is going wrong and give us some good hints about where investments will improve the model performance. For example, we see that our model in blue does track the seasonality of Dengue cases. However, the timing of the seasonality of our predictions has a mismatch with the actual results. One potential reason for this is that our features don't look far enough into the past--that is to say, we are asking to predict cases at the same time as we are measuring percipitation. Because dengue is misquito born, and the misquito lifecycle depends on water, we need to take both the life of a misquito and the time between infection and symptoms into account when modeling dengue. This is a critical avenue to explore when improving this model.
# 
# The other important error is that our predictions are relatively consistent--we miss the spikes that are large outbreaks. One reason is that we don't take into account the contagiousness of dengue. A possible way to account for this is to build a model that progressively predicts a new value while taking into account the previous prediction. By training on the dengue outbreaks and then using the predicted number of patients in the week before, we can start to model this time dependence that the current model misses.
# 
# So, we know we're not going to win this thing, but let's submit the model anyway!

sj_test_df, iq_test_df = load_data(RAW_DATA_DIR / "dengue_features_test.csv")

print("sj_test_df", sj_test_df.shape)
print("iq_test_df", iq_test_df.shape)


sj_test_preds = sj_pipeline.predict(sj_test_df)
iq_test_preds = iq_pipeline.predict(iq_test_df)


sj_submission_df, iq_submission_df = load_data(RAW_DATA_DIR / "submission_format.csv")

sj_submission_df["total_cases"] = sj_test_preds.astype(int)
iq_submission_df["total_cases"] = iq_test_preds.astype(int)

submission_df = pd.concat({
    "sj": sj_submission_df, 
    "iq": iq_submission_df
}, names=["city"])

# submission_df.to_csv("")

submission_df.head()


# ![](https://community.drivendata.org/uploads/default/original/1X/7af03e4997e8487057a77f9022691b9e9cb525f7.png)
# 
# Alright, it's a start! To build your own model you can grab this notebook [from our benchmarks repo](https://github.com/drivendata/benchmarks).
# 
# Good luck, and enjoy!
