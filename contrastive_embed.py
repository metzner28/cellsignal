# %%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.manifold import TSNE
import seaborn as sns
from helpers_contrastive_v2 import *

# %%
embeddings_train = pd.read_csv("contrastive/contrastive_embeddings.csv")
projections_train = pd.read_csv("contrastive/contrastive_projections.csv")

embeddings_test = pd.read_csv("contrastive/test_embeddings.csv")
projections_test = pd.read_csv("contrastive/test_projections.csv")

### embeddings
# %%
# trying vanilla knn
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(embeddings_train.iloc[:,:512], embeddings_train.iloc[:,513])
knn.score(embeddings_test.iloc[:,:512], embeddings_test.iloc[:,513])

# %%
predicted_labels = knn.predict(embeddings_test.iloc[:,:512])
result = pd.DataFrame(np.column_stack((embeddings_test["label"], predicted_labels)))
result.columns = ["true", "predicted"]
result["correct"] = result.apply(lambda df: df["predicted"] == df["true"], axis = 1)
np.sum(result["correct"]) / len(result)

# %%
# trying NCA
nca = NeighborhoodComponentsAnalysis(n_components = 10)
nca.fit(embeddings_train.iloc[:,:512], embeddings_train.iloc[:,513])

# %%
knn.fit(nca.transform(embeddings_train.iloc[:,:512]), embeddings_train.iloc[:,513])
knn.score(nca.transform(embeddings_test.iloc[:,:512]), embeddings_test.iloc[:,513])

# %%
tsne = TSNE()
out = tsne.fit_transform(nca.transform(embeddings_test.iloc[:,:512]))

# %%
true_labels = pd.DataFrame(embeddings_test.iloc[:,513])
true_labels.columns = ["sirna_id"]
pclasses = pd.read_csv('../sirna_functions.csv')

# %%
pclass_labels = true_labels.join(pclasses.set_index("sirna_id"), on = 'sirna_id')
out = pd.DataFrame(out)
out.columns = ["tsne_x", "tsne_y"]
tsne_labeled = out.join(pclass_labels)
df_final = tsne_labeled[tsne_labeled["parent"].notna()]

# %%
df_final.to_csv("contrastive_tsne.csv", index = False)

# %%
# tsne on raw embeddings bc why not
out_raw = pd.DataFrame(tsne.fit_transform(embeddings_test.iloc[:,:512]))
out_raw.columns = ["tsne_x", "tsne_y"]
tsne_raw_labeled = out_raw.join(pclass_labels)
df_raw_final = tsne_raw_labeled[tsne_raw_labeled["parent"].notna()]
df_raw_final.to_csv("contrastive/contrastive_tsne_raw.csv", index = False)

# %%
tsne_exp_labeled = tsne_raw_labeled.join(pd.DataFrame(embeddings_test.iloc[:,512]))
df_exp_final = tsne_exp_labeled[tsne_exp_labeled["parent"].notna()]
df_exp_final.to_csv("contrastive/contrastive_tsne_exp.csv", index = False)