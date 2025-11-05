import re
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Caricamento dataset
dataset = pd.read_csv("FinalDataset.csv")

# Preprocessing
all_stop = stopwords.words("english")
if "not" in all_stop:
    all_stop.remove("not")
mbti_types = [
    "intj", "infj", "entp", "enfp", "intp", "istp",
    "entj", "estj", "isfp", "infp", "enfj", "esfj",
    "istj", "isfj", "esfp", "estp"
]
all_stop += mbti_types
lemmatizer = WordNetLemmatizer()

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def preprocess_text(phrase):
    text = remove_urls(phrase)
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in all_stop and len(w) > 2]
    return " ".join(words)

dataset["modified_post"] = dataset["posts"].apply(preprocess_text)

# TF-IDF avanzato
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(dataset["modified_post"]).toarray()
y = dataset["type"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Modelli
models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=30),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=30),
    'LinearSVM': SVC(kernel='linear', probability=True, max_iter=2000, random_state=30),
    'MultinomialNB': MultinomialNB()
}
# Valutazione e salvataggio per singolo modello
for name, model in models.items():
    print(f"\n>>> Inizio training modello: {name}")
    t_start = time.time()

    # Applico SMOTE separatamente per monitoraggio
    print(" - Applico SMOTE...")
    smote = SMOTE(random_state=30)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f" - SMOTE completato. Shape dati: {X_train_bal.shape}")

    # Training modello
    print(" - Inizio fitting modello...")
    model.fit(X_train_bal, y_train_bal)
    print(" - Fine fitting modello. Inizio predizione...")

    y_pred = model.predict(X_test)
    t_end = time.time()

    # Salva i risultati
    results = {
        'Model': [name],
        'Accuracy': [accuracy_score(y_test, y_pred)],
        'Precision': [precision_score(y_test, y_pred, average='weighted', zero_division=0)],
        'Recall': [recall_score(y_test, y_pred, average='weighted', zero_division=0)],
        'F1 Score': [f1_score(y_test, y_pred, average='weighted', zero_division=0)],
        'Training Time (s)': [t_end - t_start]
    }

    df_model = pd.DataFrame(results)
    df_model.to_csv(f"{name}_Results.csv", index=False)
    print(f"<<< Fine modello: {name} - Precision: {results['Precision'][0]:.3f} - Tempo: {results['Training Time (s)'][0]:.1f}s")
    print(f"Risultati salvati in {name}_Results.csv")