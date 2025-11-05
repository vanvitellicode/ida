import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Caricamento e preprocessamento dei dati
df = pd.read_csv(r"FinalDataset.csv")
df = df.drop(columns=['posts'], errors='ignore')
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns='type')
y = df['type']

# Definizione dei modelli
models = {
    'Random Forest': RandomForestClassifier(n_estimators=150, criterion='gini', random_state=20),
    'AdaBoost': AdaBoostClassifier(n_estimators=150, learning_rate=1.0, random_state=20),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=20),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=20),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=20)
}

# Metriche per la cross-validation
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro', zero_division=0),
    'recall': make_scorer(recall_score, average='macro', zero_division=0),
    'f1': make_scorer(f1_score, average='macro', zero_division=0)
}

# Cross-validation stratificata a 10 fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=20)
results = []

for name, model in models.items():
    # cross_validate restituisce anche 'fit_time' e 'score_time'
    cv_res = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score='raise',
        n_jobs=-1
    )
    fit_times = cv_res['fit_time']
    score_times = cv_res['score_time']
    total_times = fit_times + score_times

    for fold in range(len(fit_times)):
        results.append({
            'Model': name,
            'Fold': fold + 1,
            'Accuracy': cv_res['test_accuracy'][fold],
            'Precision': cv_res['test_precision'][fold],
            'Recall': cv_res['test_recall'][fold],
            'F1 Score': cv_res['test_f1'][fold],
            'Fit Time (s)': fit_times[fold],
            'Score Time (s)': score_times[fold],
            'Total Time (s)': total_times[fold]
        })

# Creazione dei DataFrame dei risultati
df_results = pd.DataFrame(results)

# Riepilogo: media e deviazione standard per modello
summary = df_results.groupby('Model').agg({
    'Accuracy': ['mean', 'std'],
    'Precision': ['mean', 'std'],
    'Recall': ['mean', 'std'],
    'F1 Score': ['mean', 'std'],
    'Fit Time (s)': ['mean', 'std'],
    'Score Time (s)': ['mean', 'std'],
    'Total Time (s)': ['mean', 'std']
}).reset_index()

# Visualizzazione dei risultati
print("Risultati per ciascun fold:")
print(df_results)
print("\nMedia e deviazione standard per modello:")
print(summary)

# Salvataggio su CSV
df_results.to_csv('cv_results_per_fold_with_times.csv', index=False)
summary.to_csv('cv_summary_with_times.csv', index=False)