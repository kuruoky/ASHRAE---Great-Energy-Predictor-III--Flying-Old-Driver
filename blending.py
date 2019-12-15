import gc
import os
from pathlib import Path
import random
import sys
import itertools
from tqdm import tqdm_notebook as tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.metrics import mean_squared_error


from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


train_df = pd.read_feather('train.feather')
test_df = pd.read_feather('test.feather')

building_meta_df = pd.read_feather('building_metadata.feather')

leak_df = pd.read_feather('leak.feather')

leak_df.fillna(0, inplace=True)
leak_df = leak_df[(leak_df.timestamp.dt.year > 2016) & (leak_df.timestamp.dt.year < 2019)]
leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0
leak_df = leak_df[leak_df.building_id!=245]

sample_submission = pd.read_feather('sample_submission.feather')
sample_submission1 = pd.read_csv('1.csv', index_col=0)
sample_submission2 = pd.read_csv('2.csv', index_col=0)
sample_submission3 = pd.read_csv('6.csv', index_col=0)

test_df['pred1'] = sample_submission1.meter_reading
test_df['pred2'] = sample_submission2.meter_reading
test_df['pred3'] = sample_submission3.meter_reading

test_df.loc[test_df.pred3<0, 'pred3'] = 0


leak_df = leak_df.merge(test_df[['building_id', 'meter', 'timestamp', 'pred1', 'pred2', 'pred3', 'row_id']], left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
leak_df = leak_df.merge(building_meta_df[['building_id', 'site_id']], on='building_id', how='left')


leak_df['pred1_l1p'] = np.log1p(leak_df.pred1)
leak_df['pred2_l1p'] = np.log1p(leak_df.pred2)
leak_df['pred3_l1p'] = np.log1p(leak_df.pred3)
leak_df['meter_reading_l1p'] = np.log1p(leak_df.meter_reading)


sns.distplot(leak_df.pred1_l1p)
sns.distplot(leak_df.meter_reading_l1p)

leak_score = np.sqrt(mean_squared_error(leak_df.pred1_l1p, leak_df.meter_reading_l1p))
print ('score1=', leak_score)



sns.distplot(leak_df.pred2_l1p)
sns.distplot(leak_df.meter_reading_l1p)

leak_score = np.sqrt(mean_squared_error(leak_df.pred2_l1p, leak_df.meter_reading_l1p))
print ('score2=', leak_score)

sns.distplot(leak_df.pred3_l1p)
sns.distplot(leak_df.meter_reading_l1p)

leak_score = np.sqrt(mean_squared_error(leak_df.pred3_l1p, leak_df.meter_reading_l1p))
print ('score3=', leak_score)

leak_df['mean_pred'] = np.mean(leak_df[['pred1', 'pred2', 'pred3']].values, axis=1)
leak_df['mean_pred_l1p'] = np.log1p(leak_df.mean_pred)
leak_score = np.sqrt(mean_squared_error(leak_df.mean_pred_l1p, leak_df.meter_reading_l1p))


sns.distplot(leak_df.mean_pred_l1p)
sns.distplot(leak_df.meter_reading_l1p)

print ('mean score=', leak_score)


leak_df['median_pred'] = np.median(leak_df[['pred1', 'pred2', 'pred3']].values, axis=1)
leak_df['median_pred_l1p'] = np.log1p(leak_df.median_pred)
leak_score = np.sqrt(mean_squared_error(leak_df.median_pred_l1p, leak_df.meter_reading_l1p))

sns.distplot(leak_df.median_pred_l1p)
sns.distplot(leak_df.meter_reading_l1p)

print ('meadian score=', leak_score)



all_combinations = list(np.linspace(0.1,0.7,50))

l = [all_combinations, all_combinations, all_combinations]


all_l = list(itertools.product(*l)) + list(itertools.product(*reversed(l)))


filtered_combis = [l for l in all_l if l[0] + l[1] + l[2] > 0.95 and l[0] + l[1] + l[2] < 1.05]

best_combi = []  # of the form (i, score)
for i, combi in enumerate(filtered_combis):
    print("Now at: " + str(i) + " out of " + str(len(filtered_combis))) # uncomment to view iterations
    score1 = combi[0]
    score2 = combi[1]
    score3 = combi[2]
    v = score1 * leak_df['pred1'].values + score2 * leak_df['pred3'].values + score3 * leak_df['pred2'].values
    vl1p = np.log1p(v)
    curr_score = np.sqrt(mean_squared_error(vl1p, leak_df.meter_reading_l1p))

    if best_combi:
        prev_score = best_combi[0][1]
        if curr_score < prev_score:
            best_combi[:] = []
            best_combi += [(i, curr_score)]
    else:
        best_combi += [(i, curr_score)]

score = best_combi[0][1]
print(score)

final_combi = filtered_combis[best_combi[0][0]]
w1 = final_combi[0]
w2 = final_combi[1]
w3 = final_combi[2]
print("The weights are: w1=" + str(w1) + ", w2=" + str(w2) + ", w3=" + str(w3))

sample_submission['meter_reading'] = w1 * test_df.pred1 +  w2 * test_df.pred3  + w3 * test_df.pred2
sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0

leak_df = leak_df[['meter_reading', 'row_id']].set_index('row_id').dropna()
sample_submission.loc[leak_df.index, 'meter_reading'] = leak_df['meter_reading']



sample_submission.to_csv('blend4.csv', index=False, float_format='%.4f')

