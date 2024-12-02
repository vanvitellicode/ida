import os

def esplora_struttura_completa(percorso, livello_massimo=5):
    """
    :param percorso: P.
    :param livello_massimo: L.
    """
    def esplora_cartella(percorso, livello):
        indentazione = ' ' * 4 * livello
        elementi = sorted(os.listdir(percorso))
        cartelle = [e for e in elementi if os.path.isdir(os.path.join(percorso, e))]
        file = [e for e in elementi if os.path.isfile(os.path.join(percorso, e))]
        
        for i, cartella in enumerate(cartelle):
            if i == livello_massimo:
                print(f"{indentazione}...")
                break
            print(f"{indentazione}{cartella}/")
            esplora_cartella(os.path.join(percorso, cartella), livello + 1)
        
        if len(file) > livello_massimo:
            print(f"{indentazione}{file[0]}")
            print(f"{indentazione}...")
            print(f"{indentazione}{file[-1]}")
        else:
            for nome_file in file:
                print(f"{indentazione}{nome_file}")

    print(f"{os.path.basename(percorso)}/")
    esplora_cartella(percorso, 1)

if __name__ == "__main__":
    percorso = input("Inserisci il percorso della cartella da esplorare: ")
    if os.path.exists(percorso):
        esplora_struttura_completa(percorso)
    else:
        print("Il percorso specificato non esiste.")