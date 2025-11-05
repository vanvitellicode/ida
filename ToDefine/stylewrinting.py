import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

def extract_user_features(df):
    user_groups = df.groupby('user_id')
    rows = []

    for user_id, group in user_groups:
        texts = group['text'].dropna().tolist()
        full_text = ' '.join(texts)
        words = full_text.split()
        n_words = len(words)
        n_unique = len(set(words))

        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        punct_density = sum(1 for c in full_text if c in '.,;:!?') / len(full_text) if full_text else 0
        caps_ratio = sum(1 for c in full_text if c.isupper()) / len(full_text) if full_text else 0
        tweet_rate = len(group) / ((group['created_at'].max() - group['created_at'].min()).total_seconds() / 3600 + 1)
        ttr = n_unique / n_words if n_words else 0

        rows.append({
            'user_id': user_id,
            'avg_word_len': avg_word_len,
            'punct_density': punct_density,
            'caps_ratio': caps_ratio,
            'tweet_rate': tweet_rate,
            'type_token_ratio': ttr
        })

    return pd.DataFrame(rows)

def cluster_users(user_features, k=4):
    X = user_features.drop(columns=['user_id'])
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42)
    user_features['cluster'] = km.fit_predict(X_scaled)
    return user_features

# Esempio d'uso
df = pd.read_csv("charliehebdo-all-rnr-threads_rumours.csv", parse_dates=['created_at'])
df = df[df['tweet_type'] == 'reaction']

user_features = extract_user_features(df)
user_clusters = cluster_users(user_features, k=4)

# Join back to reactions
df = df.merge(user_clusters[['user_id', 'cluster']], on='user_id', how='left')

# Ora puoi descrivere i cluster e calcolare belief/disbelief per ciascuno