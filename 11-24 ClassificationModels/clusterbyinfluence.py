import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

for i in range(1, 5):
    if i == 1:
        data = pd.read_csv("/Users/gennarojuniorpezzullo/Desktop/Pheme Dataset/tables_divided/non-rumours/all_non_rumours_reactions.csv") 
    elif i == 2:
        data = pd.read_csv("/Users/gennarojuniorpezzullo/Desktop/Pheme Dataset/tables_divided/non-rumours/all_non_rumours_sources.csv")
    elif i == 3:
        data = pd.read_csv("/Users/gennarojuniorpezzullo/Desktop/Pheme Dataset/tables_divided/rumours/all_rumours_reactions.csv")  # Sostituisci con il percorso effettivo
    elif i == 4:
        data = pd.read_csv("/Users/gennarojuniorpezzullo/Desktop/Pheme Dataset/tables_divided/rumours/all_rumours_sources.csv")
    data = data[['user_screen_name', 'text', 'user_followers_count', 'user_friends_count', 'user_statuses_count', 'retweet_count', 'favorite_count']].dropna()

    data['engagement_rate'] = (data['retweet_count'] + data['favorite_count']) / (data['user_followers_count'] + 1)
    data['follower_friend_ratio'] = data['user_followers_count'] / (data['user_friends_count'] + 1)
    scaler = StandardScaler()
    features = data[['engagement_rate', 'follower_friend_ratio', 'user_followers_count', 'user_statuses_count']]
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['number_of_cluster'] = kmeans.fit_predict(scaled_features)
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=data['number_of_cluster'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Cluster')
    plt.title("Clustering degli utenti")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()

    output_data = data[['user_screen_name', 'text', 'number_of_cluster']].rename(columns={'user_screen_name': 'screen_name'})

    if i == 1:
        output_data.to_csv("all_non_rumours_reactions_clustered_i_a_p.csv", index=False)
    elif i == 2:
        output_data.to_csv("all_non_rumours_sources_clustered_i_a_p.csv", index=False)
    elif i == 3:
        output_data.to_csv("all_rumours_reactions_clustered_i_a_p.csv", index=False)
    elif i == 4:
        output_data.to_csv("all_rumours_sources_clustered_i_a_p.csv", index=False)
    
    print("Tabella con clustering salvata come clustered_users.csv")