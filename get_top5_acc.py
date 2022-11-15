# %%
import numpy as np
import pandas as pd

# %%
df_base = pd.read_csv("../results/preds/resnet18_baseline_mdfinal_preds.csv")
df_focal = pd.read_csv("../results/preds/resnet18_focal_mdfinal_preds.csv")

# %%
top5_base = np.argsort(df_base.iloc[:,:1139].to_numpy(), axis = 1)[:,:5]
df_top5_base = pd.DataFrame(np.column_stack((top5_base, df_base.iloc[:,1139:].to_numpy())))
df_top5_base.to_csv("resnet18_top5_mdfinal.csv", index = False)

# %%
top5_focal = np.argsort(df_focal.iloc[:,:1139].to_numpy(), axis = 1)[:,:5]
df_top5_focal = pd.DataFrame(np.column_stack((top5_focal, df_focal.iloc[:,1139:].to_numpy())))
df_top5_focal.to_csv("resnet18_focal_top5_mdfinal.csv", index = False)
