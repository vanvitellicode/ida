import os
import pandas as pd
import matplotlib.pyplot as plt

def analizza_e_visualizza_luoghi(cartella_tables):
    """
    Analizza i luoghi nei file CSV presenti in 'rumours' e 'non-rumours' e genera grafici con le percentuali di luoghi.
    
    :param cartella_tables: Percorso della directory principale contenente 'rumours' e 'non-rumours'.
    """
    # Cartelle di input
    rumours_dir = os.path.join(cartella_tables, "rumours")
    non_rumours_dir = os.path.join(cartella_tables, "non-rumours")

    # Funzione per elaborare file CSV in una directory
    def elabora_luoghi(directory):
        source_places = []
        reaction_places = []

        for file in os.listdir(directory):
            if file.endswith(".csv"):
                file_path = os.path.join(directory, file)
                df = pd.read_csv(file_path)

                # Filtra 'source-tweets' e 'reactions'
                sources = df[df['type'] == 'source-tweet']
                reactions = df[df['type'] == 'reaction']

                # Estrai i luoghi
                source_places += sources['place_full_name'].dropna().tolist()
                reaction_places += reactions['place_full_name'].dropna().tolist()

        return source_places, reaction_places

    # Elaborazione dei file
    rumours_source_places, rumours_reaction_places = elabora_luoghi(rumours_dir)
    non_rumours_source_places, non_rumours_reaction_places = elabora_luoghi(non_rumours_dir)

    # Funzione per calcolare le percentuali
    def calcola_percentuali(lista):
        df = pd.DataFrame(lista, columns=['place'])
        counts = df['place'].value_counts(normalize=True) * 100
        return counts

    # Funzione per creare grafici
    def crea_grafico(data, titolo, output_file):
        data = data.head(10)  # Prendi i primi 10 luoghi più frequenti
        data.plot(kind='bar', figsize=(10, 6))
        plt.title(titolo)
        plt.ylabel("Percentuale (%)")
        plt.xlabel("Luogo")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    # Calcolo delle percentuali
    all_source_places = rumours_source_places + non_rumours_source_places
    all_reaction_places = rumours_reaction_places + non_rumours_reaction_places

    source_percentuali = calcola_percentuali(all_source_places)
    reaction_percentuali = calcola_percentuali(all_reaction_places)

    rumours_source_percentuali = calcola_percentuali(rumours_source_places)
    rumours_reaction_percentuali = calcola_percentuali(rumours_reaction_places)

    non_rumours_source_percentuali = calcola_percentuali(non_rumours_source_places)
    non_rumours_reaction_percentuali = calcola_percentuali(non_rumours_reaction_places)

    # Creazione dei grafici
    output_dir = "grafici"
    os.makedirs(output_dir, exist_ok=True)

    crea_grafico(source_percentuali, "Luoghi più frequenti (Source - Tutti)", os.path.join(output_dir, "source_tutti.png"))
    crea_grafico(reaction_percentuali, "Luoghi più frequenti (Reactions - Tutti)", os.path.join(output_dir, "reaction_tutti.png"))
    crea_grafico(rumours_source_percentuali, "Luoghi più frequenti (Source - Rumours)", os.path.join(output_dir, "source_rumours.png"))
    crea_grafico(rumours_reaction_percentuali, "Luoghi più frequenti (Reactions - Rumours)", os.path.join(output_dir, "reaction_rumours.png"))
    crea_grafico(non_rumours_source_percentuali, "Luoghi più frequenti (Source - Non-Rumours)", os.path.join(output_dir, "source_non_rumours.png"))
    crea_grafico(non_rumours_reaction_percentuali, "Luoghi più frequenti (Reactions - Non-Rumours)", os.path.join(output_dir, "reaction_non_rumours.png"))

    print(f"Grafici salvati nella directory '{output_dir}'")

if __name__ == "__main__":
    cartella_tables = "tables"
    if os.path.exists(cartella_tables):
        analizza_e_visualizza_luoghi(cartella_tables)
    else:
        print("La cartella 'tables' non esiste.")