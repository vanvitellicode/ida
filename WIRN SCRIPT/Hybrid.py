import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = stopwords.words("english")
if "not" in stop_words:
    stop_words.remove("not")
mbti_types = [
    "intj", "infj", "entp", "enfp", "intp", "istp",
    "entj", "estj", "isfp", "infp", "enfj", "esfj",
    "istj", "isfj", "esfp", "estp"
]
stop_words += mbti_types
lemmatizer = WordNetLemmatizer()

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def preprocess_text(text):
    text = remove_urls(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# Load dataset
df = pd.read_csv("FinalDataset.csv")
df["modified_post"] = df["posts"].apply(preprocess_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_text = tfidf.fit_transform(df["modified_post"]).toarray()

# Preprocessing metadata
metadata = df.drop(columns=["posts", "modified_post"], errors='ignore').copy()
for col in metadata.select_dtypes(include='object').columns:
    metadata[col] = LabelEncoder().fit_transform(metadata[col])
X_meta = metadata.drop(columns=["type"])
y = metadata["type"]

# Concatenazione TF-IDF e metadati
X_combined = np.hstack([X_text, X_meta.values])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, stratify=y, test_size=0.2, random_state=42)

# Modelli
models = {
    'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced'),
    'LogisticRegression': LogisticRegression(solver='liblinear', max_iter=3000, random_state=42, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=300, random_state=42),
    'LinearSVM': SVC(kernel='linear', probability=True, max_iter=2000, random_state=42),
    'MultinomialNB': MultinomialNB()
}

# Valutazione e salvataggio
for name, model in models.items():
    print(f"\n>>> Training modello ibrido: {name}")
    start = time.time()
    
    pipe = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    end = time.time()
    results = {
        'Model': [name],
        'Accuracy': [accuracy_score(y_test, y_pred)],
        'Precision': [precision_score(y_test, y_pred, average='weighted', zero_division=0)],
        'Recall': [recall_score(y_test, y_pred, average='weighted', zero_division=0)],
        'F1 Score': [f1_score(y_test, y_pred, average='weighted', zero_division=0)],
        'Training Time (s)': [end - start]
    }
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"Hybrid_{name}_Results.csv", index=False)
    
    print(f"<<< Fine modello ibrido: {name} - F1: {results['F1 Score'][0]:.3f} - Tempo: {results['Training Time (s)'][0]:.1f}s")
    print(f"Risultati salvati in Hybrid_{name}_Results.csv")