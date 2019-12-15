import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import math

import optuna

train_df = pd.read_csv('./train.csv')
weather_train_df = pd.read_csv('./weather_train.csv')
test_df = pd.read_csv('./test.csv')
weather_test_df = pd.read_csv('./weather_test.csv')
building_meta_df = pd.read_csv('./building_metadata.csv')
sample_submission = pd.read_csv('./sample_submission.csv')

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])

train_df.to_feather('train.feather')
test_df.to_feather('test.feather')
weather_train_df.to_feather('weather_train.feather')
weather_test_df.to_feather('weather_test.feather')
building_meta_df.to_feather('building_metadata.feather')
sample_submission.to_feather('sample_submission.feather')


train_df = pd.read_feather('train.feather')
weather_train_df = pd.read_feather('weather_train.feather')
test_df = pd.read_feather('test.feather')
weather_test_df = pd.read_feather('weather_test.feather')
building_meta_df = pd.read_feather('building_metadata.feather')
sample_submission = pd.read_feather('sample_submission.feather')






















