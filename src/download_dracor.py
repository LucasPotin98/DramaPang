import pandas as pd
import os
import requests
from io import StringIO

# Charger le CSV global avec les métadonnées
metadata_df = pd.read_csv("fredracor.csv")

# Dossiers de sortie
NETWORK_DIR = "dracor_networks_fre"
CHARACTER_DIR = "dracor_characters_fre"
os.makedirs(NETWORK_DIR, exist_ok=True)
os.makedirs(CHARACTER_DIR, exist_ok=True)

BASE = "https://dracor.org/api/v1"

for idx, row in metadata_df.iterrows():
    play_id = row["name"]
    print(f"🔄 Téléchargement : {play_id}")

    # 1. Récupérer réseau
    network_url = f"{BASE}/corpora/fre/plays/{play_id}/networkdata/csv"
    try:
        net_resp = requests.get(network_url)
        net_text = net_resp.text.strip()
        if not net_text:
            print(f"⚠️ Réseau vide pour {play_id}")
            continue

        df_net = pd.read_csv(StringIO(net_text))
        expected_cols = {"Source", "Target", "Weight"}
        if not expected_cols.issubset(set(df_net.columns)):
            print(f"⚠️ Colonnes manquantes dans réseau de {play_id}")
            continue

        # Sauvegarde réseau
        df_net.to_csv(f"{NETWORK_DIR}/{play_id}.csv", index=False)
        print(f"✅ Réseau sauvegardé pour {play_id}")

        # 2. Récupérer personnages
        char_url = f"{BASE}/corpora/fre/plays/{play_id}/characters"
        char_resp = requests.get(char_url)
        if char_resp.status_code != 200:
            print(f"❌ Échec personnages pour {play_id}")
            continue

        df_char = pd.json_normalize(char_resp.json())
        df_char.to_csv(f"{CHARACTER_DIR}/{play_id}.csv", index=False)
        print(f"✅ Personnages sauvegardés pour {play_id}")

    except Exception as e:
        print(f"❌ Erreur pour {play_id} : {e}")
