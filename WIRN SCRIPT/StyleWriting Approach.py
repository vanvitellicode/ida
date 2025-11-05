import re
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scarica risorse NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 1) Carica dataset
dataset = pd.read_csv("FinalDataset.csv")

# 2) Preprocessing testo
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

# 3) TF-IDF vectorization (3000 feature)
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(dataset["modified_post"]).toarray()
y = dataset["type"]

# 4) Modelli selezionati con class_weight (dove utile)
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
    'LogisticRegression': LogisticRegression(solver='liblinear', max_iter=1000, random_state=42, class_weight='balanced')
}

# 5) Cross-validation 4-fold
cv_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
scoring = {
    'accuracy':     'accuracy',
    'precision':    'precision_weighted',
    'recall':       'recall_weighted',
    'f1':           'f1_weighted'
}

# 6) Esecuzione modelli e salvataggio CSV
for name, mdl in models.items():
    print(f"\n>>> Inizio esecuzione modello: {name}")
    t_start_model = time.time()

    pipe = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', mdl)
    ])

    cv_res = cross_validate(
        estimator=pipe,
        X=X, y=y,
        cv=cv_outer,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    t_end_model = time.time()
    print(f"<<< Fine {name} in {t_end_model - t_start_model:.1f} secondi")

    results = []
    for fold in range(cv_outer.get_n_splits()):
        results.append({
            'Model':          name,
            'Fold':           fold + 1,
            'Accuracy':       cv_res['test_accuracy'][fold],
            'Precision':      cv_res['test_precision'][fold],
            'Recall':         cv_res['test_recall'][fold],
            'F1 Score':       cv_res['test_f1'][fold],
            'Fit Time (s)':   cv_res['fit_time'][fold],
            'Score Time (s)': cv_res['score_time'][fold],
            'Total Time (s)': cv_res['fit_time'][fold] + cv_res['score_time'][fold]
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{name}_CV_Results.csv", index=False)
    print(f"Risultati salvati in {name}_CV_Results.csv")