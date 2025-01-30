import pandas as pd
comment_id_counter = 1
all_dfs = []
for i in range(1, 10):
    df = pd.read_csv(f"Custom Dataset/Post Dataset/{i}.csv")
    df["id_dataset"] = i
    num_righe = len(df)
    df["id_commento"] = range(comment_id_counter, comment_id_counter + num_righe)
    comment_id_counter += num_righe
    df_out = df[["id_dataset", "id_commento", "Content", "ReactionsCount"]]
    all_dfs.append(df_out)
final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv("alldata.csv", index=False)