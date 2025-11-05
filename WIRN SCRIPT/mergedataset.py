import pandas as pd

df_small = pd.read_csv("StyleWritingDataset.csv") 
df_big = pd.read_csv("ClassificationDataset.csv", sep=';') 
df_big.rename(columns={'Personality': 'type'}, inplace=True)

df_augmented = df_small.copy()
extra_columns = [col for col in df_big.columns if col != 'type']
for col in extra_columns:
    df_augmented[col] = None

grouped_big = df_big.groupby('type')

group_counters = {t: 0 for t in grouped_big.groups.keys()}

for idx, row in df_augmented.iterrows():
    personality = row['type']
    if personality in grouped_big.groups:
        group = grouped_big.get_group(personality)
        counter = group_counters[personality]
        if counter < len(group):
            selected_row = group.iloc[counter]
            group_counters[personality] += 1
            for col in extra_columns:
                df_augmented.at[idx, col] = selected_row[col]
df_augmented.to_csv("dataset_arricchito.csv", index=False)