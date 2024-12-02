import pandas as pd

# Carica il file CSV e seleziona la prima colonna
df = pd.read_csv("frasi.csv", usecols=[0])

# Converte la colonna in una lista
l = df.iloc[:, 0].tolist()

# Mostra le prime righe del DataFrame
print(l[0])