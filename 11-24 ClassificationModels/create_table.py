import os
import pandas as pd

def divide_tabelle(input_dir, output_dir):
    # Percorsi delle directory rumours e non-rumours
    rumours_dir = os.path.join(input_dir, "rumours")
    non_rumours_dir = os.path.join(input_dir, "non-rumours")
    
    # Creazione della directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # File di output
    rumorsource_path = os.path.join(output_dir, "rumorsource.csv")
    rumorresponse_path = os.path.join(output_dir, "rumorresponse.csv")
    nonrumorsource_path = os.path.join(output_dir, "nonrumorsource.csv")
    nonrumorresponse_path = os.path.join(output_dir, "nonrumorresponse.csv")
    
    # Liste per raccogliere i dati
    rumorsource = []
    rumorresponse = []
    nonrumorsource = []
    nonrumorresponse = []
    
    # Funzione per elaborare una directory
    def process_directory(directory, source_list, reaction_list):
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                file_path = os.path.join(directory, file)
                df = pd.read_csv(file_path)
                
                # Separare source-tweets e reactions
                sources = df[df['type'] == 'source-tweet']
                reactions = df[df['type'] == 'reaction']
                
                # Aggiungere alle rispettive liste
                source_list.append(sources)
                reaction_list.append(reactions)
    
    # Elaborazione delle directory
    process_directory(rumours_dir, rumorsource, rumorresponse)
    process_directory(non_rumours_dir, nonrumorsource, nonrumorresponse)
    
    # Concatenazione dei dati
    rumorsource_df = pd.concat(rumorsource, ignore_index=True)
    rumorresponse_df = pd.concat(rumorresponse, ignore_index=True)
    nonrumorsource_df = pd.concat(nonrumorsource, ignore_index=True)
    nonrumorresponse_df = pd.concat(nonrumorresponse, ignore_index=True)
    
    # Salvataggio nei file CSV
    rumorsource_df.to_csv(rumorsource_path, index=False)
    rumorresponse_df.to_csv(rumorresponse_path, index=False)
    nonrumorsource_df.to_csv(nonrumorsource_path, index=False)
    nonrumorresponse_df.to_csv(nonrumorresponse_path, index=False)
    
    print(f"File salvati nella directory: {output_dir}")

if __name__ == "__main__":
    input_dir = "tables"  # Sostituisci con il percorso della directory di input
    output_dir = "tabledivided"
    divide_tabelle(input_dir, output_dir)