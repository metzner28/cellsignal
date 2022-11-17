# %%
import numpy as np
import pandas as pd
import sys

# %%
def get_top5(df):
    
    '''
    Input: dataframe of shape (n_obs, n_classes) with probabilities of each class for each obs
    Output: dataframe of (n_obs, 3), where col 1 is 1 if obs was correctly classified else 0; col 2 is 1 if obs was correctly classified or if the correct label was predicted with probability in the top 5 for that obs
    '''

    top5 = np.fliplr(np.argsort(df.iloc[:,:1139].to_numpy(), axis = 1))[:,:5]
    df_top5 = pd.DataFrame(np.column_stack((top5, df.iloc[:,1139:].to_numpy())))

    df_top5.columns = ["max_pred", "pred2", "pred3", "pred4", "pred5", "exp", "label"]

    # there has to be a better way to do this
    df_top5["top1"] = df_top5.apply(lambda df: 1 if df["max_pred"] == df["label"] else 0, axis = 1)
    df_top5["top2"] = df_top5.apply(lambda df: 1 if df["pred2"] == df["label"] else 0, axis = 1)
    df_top5["top3"] = df_top5.apply(lambda df: 1 if df["pred3"] == df["label"] else 0, axis = 1)
    df_top5["top4"] = df_top5.apply(lambda df: 1 if df["pred4"] == df["label"] else 0, axis = 1)
    df_top5["top5th"] = df_top5.apply(lambda df: 1 if df["pred5"] == df["label"] else 0, axis = 1)

    df_top5["top5"] = df_top5.apply(lambda df: df["top1"] + df["top2"] + df["top3"] + df["top4"] + df["top5th"], axis = 1)

    df_top5 = df_top5.drop(columns = ["top2", "top3", "top4", "top5th"])

    return df_top5

# %%
if __name__ == "__main__":

    try:
        df = pd.read_csv(sys.argv[1])
        df_out = get_top5(df)
        df_string = sys.argv[1].split("/")[-1].split(".")[0]
        df_out.to_csv(f"{df_string}_top5.csv", index = False)

    except:
        raise ValueError("issue with input predictions dataframe")