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


def create_lag_features(df, window):
    """
    Creating lag-based features looking back in time.
    """

    feature_cols = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr", "sea_level_pressure",
                    "wind_direction", "wind_speed"]

    df_site = df.groupby("site_id")

    df_rolled = df_site[feature_cols].rolling(window=window, min_periods=0)

    df_mean = df_rolled.mean().reset_index().astype(np.float16)
    df_median = df_rolled.median().reset_index().astype(np.float16)
    df_min = df_rolled.min().reset_index().astype(np.float16)
    df_max = df_rolled.max().reset_index().astype(np.float16)
    df_std = df_rolled.std().reset_index().astype(np.float16)
    df_skew = df_rolled.skew().reset_index().astype(np.float16)

    for feature in feature_cols:
        df[f"{feature}_mean_lag{window}"] = df_mean[feature]
        df[f"{feature}_median_lag{window}"] = df_median[feature]
        df[f"{feature}_min_lag{window}"] = df_min[feature]
        df[f"{feature}_max_lag{window}"] = df_max[feature]
        df[f"{feature}_std_lag{window}"] = df_std[feature]
        df[f"{feature}_skew_lag{window}"] = df_std[feature]

    return df


dict_norm = {}

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')


building = pd.read_csv('./building_metadata.csv')
label = LabelEncoder()
building.primary_use = label.fit_transform(building.primary_use)


weather_train = pd.read_csv('./weather_train.csv')
weather_test = pd.read_csv('./weather_test.csv')

weather_train = weather_train.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))
weather_test = weather_test.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))

df_train['meter_reading'] = df_train['meter_reading'] + 10

for i in range(1449):
    for j in range(4):
        index = df_train[(df_train['building_id'] == i) & (df_train['meter'] == j)].index
        if np.shape(index)[0] < 2:
            continue
        if (np.shape(index)[0] < 731) & (np.shape(index)[0] > 2):
            df_train.loc[index, 'window_mean'] = df_train.loc[index, 'meter_reading'].rolling(np.shape(index)[0]).mean()
            df_matrix = df_train.loc[index, 'window_mean'].tolist()
            mean = df_matrix[np.shape(index)[0] - 1]
            for k in range(np.shape(index)[0]):
                df_matrix[k] = mean
            df_train.loc[index, 'window_mean'] = df_matrix
            dict_norm[str(i) + '_' + str(j)] = df_matrix
            df_reading = df_train.loc[index, 'meter_reading'].tolist()
            for l in range(np.shape(df_reading)[0]):
                df_reading[l] = (df_reading[l] - df_matrix[l]) / df_matrix[l]
            df_train.loc[index, 'meter_reading'] = df_reading
            continue
        df_train.loc[index,'window_mean'] = df_train.loc[index, 'meter_reading'].rolling(731).mean()
        df_matrix = df_train.loc[index,'window_mean'].tolist()
        tot = math.floor(np.shape(df_matrix)[0] / 731)
        for k in range(tot):
            mean = df_matrix[(k+1)*731 - 1]
            for w in range(k*731, (k+1)*731 - 1):
                df_matrix[w] = mean
        mean =  df_matrix[tot*731 - 1]
        for k in range(tot*731 - 1, np.shape(df_matrix)[0]):
            df_matrix[k] = mean
        df_train.loc[index, 'window_mean'] = df_matrix
        dict_norm[str(i) + '_' + str(j)] = df_matrix
        df_reading = df_train.loc[index, 'meter_reading'].tolist()
        for l in range(np.shape(df_reading)[0]):
            df_reading[l] = (df_reading[l] - df_matrix[l]) / df_matrix[l]
        df_train.loc[index, 'meter_reading'] = df_reading


df_train = df_train.merge(building, on="building_id")
df_train = df_train.merge(weather_train, on=["site_id", "timestamp"], how="left")
df_train = df_train[~((df_train.site_id==0) & (df_train.meter==0) & (df_train.building_id <= 104) & #As you can see above, this data looks weired until May 20. It is reported in this discussion by
                                                                                            # @barnwellguy that All electricity meter is 0 until May 20 for site_id == 0. I will remove these data from training data.
                            (df_train.timestamp < "2016-05-21"))]

df_train.reset_index(drop=True, inplace=True)
df_train.timestamp = pd.to_datetime(df_train.timestamp, format='%Y-%m-%d %H:%M:%S')
#df_train["log_meter_reading"] = np.log1p(df_train.meter_reading)


df_test = df_test.merge(building, on="building_id")
df_test = df_test.merge(weather_test, on=["site_id", "timestamp"], how="left")
df_test.reset_index(drop=True, inplace=True)
df_test.timestamp = pd.to_datetime(df_test.timestamp, format='%Y-%m-%d %H:%M:%S')

df_train["hour"] = df_train.timestamp.dt.hour
df_train["weekday"] = df_train.timestamp.dt.weekday

df_test["hour"] = df_test.timestamp.dt.hour
df_test["weekday"] = df_test.timestamp.dt.weekday



df_building_meter = df_train.groupby(["building_id", "meter"]).agg(mean_building_meter=("meter_reading", "mean"),
                                                             median_building_meter=("meter_reading", "median")).reset_index()

df_train = df_train.merge(df_building_meter, on=["building_id", "meter"])
df_test = df_test.merge(df_building_meter, on=["building_id", "meter"])

df_building_meter_hour = df_train.groupby(["building_id", "meter", "hour"]).agg(mean_building_meter=("meter_reading", "mean"),
                                                                     median_building_meter=("meter_reading", "median")).reset_index()

df_train = df_train.merge(df_building_meter_hour, on=["building_id", "meter", "hour"])
df_test = df_test.merge(df_building_meter_hour, on=["building_id", "meter", "hour"])


weather_train = create_lag_features(weather_train, 18)
weather_train.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr", "sea_level_pressure",
                    "wind_direction", "wind_speed"], axis=1, inplace=True)



categorical_features = [
  #  "building_id",
    "primary_use",
    "meter",
    "weekday",
    "hour"
]

all_features = [col for col in df_train.columns if col not in ["timestamp", "site_id", "meter_reading", "building_id", "window_mean"]]
#np.save('dict.npy', dict_norm)


cv = 2
models = {}
cv_scores = {"site_id": [], "cv_score": []}

for site_id in tqdm(range(16), desc="site_id"):
    print(cv, "fold CV for site_id:", site_id)
    kf = KFold(n_splits=cv, random_state=12138)
    models[site_id] = []

    X_train_site = df_train[df_train.site_id == site_id].reset_index(drop=True)
    y_train_site = X_train_site.meter_reading
    y_pred_train_site = np.zeros(X_train_site.shape[0])

    score = 0

    for fold, (train_index, valid_index) in enumerate(kf.split(X_train_site, y_train_site)):
        X_train, X_valid = X_train_site.loc[train_index, all_features], X_train_site.loc[valid_index, all_features]
        y_train, y_valid = y_train_site.iloc[train_index], y_train_site.iloc[valid_index]
        print(np.any(np.isnan(y_valid)))

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)

        watchlist = [dtrain, dvalid]

        params = {"objective": "regression",
                  "num_leaves": 41,
                  "learning_rate": 0.049,
                  "bagging_freq": 5,
                  "bagging_fraction": 0.51,
                  "feature_fraction": 0.81,
                  "metric": "rmse"
                  }

        model_lgb = lgb.train(params, train_set=dtrain, num_boost_round=999, valid_sets=watchlist, verbose_eval=101,
                              early_stopping_rounds=21)
        models[site_id].append(model_lgb)

        y_pred_valid = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
        y_pred_train_site[valid_index] = y_pred_valid

        rmse = np.sqrt(np.abs(mean_squared_error(y_valid, y_pred_valid)))
        print("Site Id:", site_id, ", Fold:", fold + 1, ", RMSE:", rmse)
        score += rmse / cv

        gc.collect()

    cv_scores["site_id"].append(site_id)
    cv_scores["cv_score"].append(score)

    print("\nSite Id:", site_id, ", CV RMSE:", np.sqrt(mean_squared_error(y_train_site, y_pred_train_site)), "\n")

lgb.plot_importance(model_lgb)
plt.show()


weather_test = create_lag_features(weather_test, 18)
weather_test.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr", "sea_level_pressure",
                    "wind_direction", "wind_speed"], axis=1, inplace=True)

weather_test.to_csv('./weathertest.csv')

df_test_sites = []

for site_id in tqdm(range(16), desc="site_id"):
    print("Preparing test data for site_id", site_id)

    X_test_site = df_test[df_test.site_id == site_id]
    weather_test_site = weather_test[weather_test.site_id == site_id]

    weather_test_site.timestamp = pd.to_datetime(weather_test_site.timestamp)
    X_test_site.timestamp = pd.to_datetime(X_test_site.timestamp)

    X_test_site = X_test_site.merge(weather_test_site, on=["site_id", "timestamp"], how="left")


    row_ids_site = X_test_site.row_id

    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])

    print("Scoring for site_id", site_id)
    for fold in range(cv):
        model_lgb = models[site_id][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)

    print("Scoring for site_id", site_id, "completed\n")
    gc.collect()





submit = pd.concat(df_test_sites)
submit.meter_reading = np.clip(submit.meter_reading, 0, a_max=None)
submit.to_csv("submission_noleak.csv", index=False)



for i in range(105):
    for j in range(4):
        index = df_test[(df_test['building_id'] == i) & (df_test['meter'] == j)].index.tolist()
        if np.shape(index)[0] < 2:
            continue
        for k in range(np.shape(index)[0]):
            submit.meter_reading[index[k]] = (submit.meter_reading[index[k]] * dict_norm[str(i) + '_' + str(j)][k]) + \
                                         dict_norm[str(i) + '_' + str(j)][k] - 10


submit.to_csv("submission_noleak.csv", index=False)
