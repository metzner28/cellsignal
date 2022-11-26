# %%
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
os.chdir("/Volumes/DRIVE/troubleshoot_results")

# %%
preds = pd.read_csv("troubleshoot_preds.csv")
embeddings = pd.read_csv("troubleshoot_embeddings.csv")
projections = pd.read_csv("troubleshoot_projections.csv")

# %%
knn = KNeighborsClassifier()
knn.fit(embeddings.iloc[:,:512], embeddings.iloc[:,513])

# %%
predicted_labels = knn.predict(embeddings.iloc[:,:512])

# %%
result = pd.DataFrame(np.column_stack((embeddings["label"], predicted_labels)))
result.columns = ["true", "predicted"]
result["correct"] = result.apply(lambda df: df["predicted"] == df["true"], axis = 1)

# %%
df_skip = embeddings.iloc[::2,:]
df_skip_train = df_skip.iloc[:9294,:]
df_skip_val = df_skip.iloc[9294:, :]

# %%
knn_skip = KNeighborsClassifier(n_neighbors = 2)
knn_skip.fit(df_skip_train.iloc[:,:512], df_skip_train.iloc[:,513])
predicted_skip = knn_skip.predict(df_skip_val.iloc[:,:512])

# %%
result_sk = pd.DataFrame(np.column_stack((df_skip_val["label"], predicted_skip)))
result_sk.columns = ["true", "predicted"]
result_sk["correct"] = result_sk.apply(lambda df: df["predicted"] == df["true"], axis = 1)
np.sum(result_sk["correct"]) / len(result_sk)