import os
import json
import pandas as pd

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '_')
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '_')
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def esplora_e_crea_tabelle_aggregate(percorso_base, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rumours_dir = os.path.join(output_dir, "rumours")
    non_rumours_dir = os.path.join(output_dir, "non-rumours")
    os.makedirs(rumours_dir, exist_ok=True)
    os.makedirs(non_rumours_dir, exist_ok=True)
    all_rumours_sources = []
    all_rumours_reactions = []
    all_non_rumours_sources = []
    all_non_rumours_reactions = []
    
    for root, dirs, files in os.walk(percorso_base):
        source_tweets = [f for f in files if f.endswith('.json') and 'source-tweet' in root]

        if "rumours" in os.path.normpath(root).split(os.sep):
            current_source_list = all_rumours_sources
            current_reaction_list = all_rumours_reactions
        elif "non-rumours" in os.path.normpath(root).split(os.sep):
            current_source_list = all_non_rumours_sources
            current_reaction_list = all_non_rumours_reactions
        else:
            continue

        for st in source_tweets:
            source_path = os.path.join(root, st)

            with open(source_path, 'r') as f:
                source_data = json.load(f)
            source_row = flatten_json({"type": "source-tweet", **source_data})
            current_source_list.append(source_row)
            reaction_dir = root.replace('source-tweet', 'reactions')
            if os.path.exists(reaction_dir):
                reaction_files = [f for f in os.listdir(reaction_dir) if f.endswith('.json')]
                for reaction_file in reaction_files:
                    reaction_path = os.path.join(reaction_dir, reaction_file)
                    with open(reaction_path, 'r') as f:
                        reaction_data = json.load(f)
                        reaction_row = flatten_json({"type": "reaction", **reaction_data})
                        current_reaction_list.append(reaction_row)
    if all_rumours_sources:
        df_rumours_sources = pd.DataFrame(all_rumours_sources)
        df_rumours_sources.to_csv(os.path.join(rumours_dir, "all_rumours_sources.csv"), index=False)
        print(f"Tabella creata: {os.path.join(rumours_dir, 'all_rumours_sources.csv')}")

    if all_rumours_reactions:
        df_rumours_reactions = pd.DataFrame(all_rumours_reactions)
        df_rumours_reactions.to_csv(os.path.join(rumours_dir, "all_rumours_reactions.csv"), index=False)
        print(f"Tabella creata: {os.path.join(rumours_dir, 'all_rumours_reactions.csv')}")

    if all_non_rumours_sources:
        df_non_rumours_sources = pd.DataFrame(all_non_rumours_sources)
        df_non_rumours_sources.to_csv(os.path.join(non_rumours_dir, "all_non_rumours_sources.csv"), index=False)
        print(f"Tabella creata: {os.path.join(non_rumours_dir, 'all_non_rumours_sources.csv')}")

    if all_non_rumours_reactions:
        df_non_rumours_reactions = pd.DataFrame(all_non_rumours_reactions)
        df_non_rumours_reactions.to_csv(os.path.join(non_rumours_dir, "all_non_rumours_reactions.csv"), index=False)
        print(f"Tabella creata: {os.path.join(non_rumours_dir, 'all_non_rumours_reactions.csv')}")

if __name__ == "__main__":
    percorso_base = "pheme-rnr-dataset"
    output_dir = "tables_divided"
    if os.path.exists(percorso_base):
        esplora_e_crea_tabelle_aggregate(percorso_base, output_dir)
        print(f"Tabelle aggregate salvate in '{output_dir}' organizzate in 'rumours' e 'non-rumours'.")
    else:
        print("Il percorso specificato non esiste.")