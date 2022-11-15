# %%
import numpy as np
import pandas as pd
import sys

# %%
def get_top5(df):

    top5 = np.fliplr(np.argsort(df.iloc[:,:1139].to_numpy(), axis = 1))[:,:5]
    
    df_top5 = pd.DataFrame(np.column_stack((top5, df.iloc[:,1139:].to_numpy())))

    df_top5.columns = ["max_pred", "pred2", "pred3", "pred4", "pred5", "exp", "label"]

    df_top5["top1"] = df_top5.apply(lambda df: 1 if df["max_pred"] == df["label"] else 0, axis = 1)

    df_top5["top2"] = df_top5.apply(lambda df: 1 if df["pred2"] == df["label"] else 0, axis = 1)

    df_top5["top3"] = df_top5.apply(lambda df: 1 if df["pred3"] == df["label"] else 0, axis = 1)

    df_top5["top4"] = df_top5.apply(lambda df: 1 if df["pred4"] == df["label"] else 0, axis = 1)

    df_top5["top5th"] = df_top5.apply(lambda df: 1 if df["pred5"] == df["label"] else 0, axis = 1)

    df_top5["top5"] = df_top5.apply(lambda df: df["top1"] + df["top2"] + df["top3"] + df["top4"] + df["top5th"], axis = 1)

    df_top5 = df_top5.drop(columns = ["top2", "top3", "top4", "top5"])

    return df_top5

if __name__ == "__main__":

    try:
        df = pd.read_csv(sys.argv[1])
        df_out = get_top5(df)
        df_string = sys.argv[1].split("/")[-1].split(".")[0]
        df_out.to_csv(f"{df_string}_top5.csv")

    except:
        raise ValueError("issue with input predictions dataframe")