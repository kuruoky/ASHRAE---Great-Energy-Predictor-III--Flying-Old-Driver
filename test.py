import pandas as pd
import numpy as np


df_test = pd.read_csv("./test.csv")
df_sample = pd.read_csv("./sample_submission.csv")
index = df_test[df_test['row_id'] <= np.max(df_sample['row_id'])].index
df_test = df_test.loc[index, :]
print(np.shape(df_sample))
print(np.shape(df_test))
submit = pd.read_csv("./submission.csv")
dict = np.load("./dict3.npy", allow_pickle=True)
dict = dict.item()
list = submit.meter_reading.tolist()




for i in range(1449):
    print(i)
    for j in range(4):
        print(j)
        index = df_test[(df_test['building_id'] == i) & (df_test['meter'] == j)].index.tolist()
        if np.shape(index)[0] < 2:
            continue
        for k in range(np.shape(index)[0]):
            k1 = k % np.shape(dict[str(i) + '_' + str(j)])[0]
            list[index[k]] = (list[index[k]] * dict[str(i) + '_' + str(j)][k1]) + \
                                         dict[str(i) + '_' + str(j)][k1] - 10



df_sample.meter_reading = list

df_sample.to_csv("./submission.csv", index=False, float_format='%.4f')






