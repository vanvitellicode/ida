import pandas as pd
import matplotlib.pyplot as plt

# Carica i dati
charlie = pd.read_csv("kpi_charliehebdo.csv")
putin = pd.read_csv("kpi_putinmissing.csv")

# Aggiunge una colonna per etichettare l'evento
charlie['event'] = 'Charlie Hebdo'
putin['event'] = 'Putin Missing'

# Unisce i dataset
df = pd.concat([charlie, putin])

# KPI da confrontare
kpi_columns = [
    'unique_reach',
    'views',
    'time_to_initial_spread_min',
    'content_lifespan_min',
    'grow_rate_tweets_per_min'
]

# Funzione per creare un grafico comparativo per ogni KPI
def plot_kpi_comparison(df, kpi_name):
    plt.figure(figsize=(12, 6))
    df_grouped = df.groupby('event')[kpi_name].describe()[['mean', 'std']]

    # Barre con errore standard
    plt.bar(df_grouped.index, df_grouped['mean'], yerr=df_grouped['std'], capsize=8)
    plt.title(f"Confronto KPI: {kpi_name}")
    plt.ylabel(kpi_name)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"kpi_comparison_{kpi_name}.png")
    plt.close()

# Genera grafici per ogni KPI
for kpi in kpi_columns:
    plot_kpi_comparison(df, kpi)

print("✅ Grafici salvati come PNG.")