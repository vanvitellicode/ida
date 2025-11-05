import re
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Classificatori
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, RidgeClassifier

# === Caricamento e Preprocessing Dataset ===
df = pd.read_csv("FinalDataset.csv")

# Pulizia del testo
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text.lower()

df["clean_post"] = df["posts"].apply(clean_text)

# TF-IDF su testo
tfidf = TfidfVectorizer(max_features=500)
X_text = tfidf.fit_transform(df["clean_post"])

# Encoding della label
le = LabelEncoder()
y = le.fit_transform(df["type"])

# === Definizione Modelli ===
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1),
    'SGD Classifier': SGDClassifier(max_iter=500, n_jobs=-1),
    'Ridge Classifier': RidgeClassifier(),
    'Multinomial NB': MultinomialNB(),
    'Bernoulli NB': BernoulliNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=20),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=20),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=20)
}

# === Metriche ===
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro', zero_division=0),
    'recall': make_scorer(recall_score, average='macro', zero_division=0),
    'f1': make_scorer(f1_score, average='macro', zero_division=0)
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=20)

# === Cross-validation ===
results = []
for name, model in models.items():
    print(f"\n>>> Esecuzione modello: {name}")
    cv_res = cross_validate(
        model, X_text, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    for fold in range(len(cv_res['fit_time'])):
        results.append({
            'Model': name,
            'Fold': fold + 1,
            'Accuracy': cv_res['test_accuracy'][fold],
            'Precision': cv_res['test_precision'][fold],
            'Recall': cv_res['test_recall'][fold],
            'F1 Score': cv_res['test_f1'][fold],
            'Fit Time (s)': cv_res['fit_time'][fold],
            'Score Time (s)': cv_res['score_time'][fold],
            'Total Time (s)': cv_res['fit_time'][fold] + cv_res['score_time'][fold]
        })

# === Salvataggio e Riepilogo ===
df_results = pd.DataFrame(results)
summary = df_results.groupby('Model').agg({
    'Accuracy': ['mean', 'std'],
    'Precision': ['mean', 'std'],
    'Recall': ['mean', 'std'],
    'F1 Score': ['mean', 'std'],
    'Fit Time (s)': ['mean', 'std'],
    'Score Time (s)': ['mean', 'std'],
    'Total Time (s)': ['mean', 'std']
}).reset_index()

# Salva su CSV
df_results.to_csv("hybrid_cv_results_per_fold.csv", index=False)
summary.to_csv("hybrid_cv_summary.csv", index=False)

# Output finale
print("\n=== Riepilogo per modello (media ± std) ===")
print(summary)