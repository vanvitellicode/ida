import random
import pandas as pd
import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import pickle
import os, sys, re, html
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
import nltk
nltk.download('wordnet')
from nltk.corpus import words
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, accuracy_score
from scipy.stats import chi2_contingency
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
font = FontProperties()
font.set_size(8)
font2 = FontProperties()
font2.set_size(8)
font2.set_weight("bold")
font3 = FontProperties()
font3.set_size(7)
model_type = os.getenv('MODEL_TYPE', 'mlp') 
print(f"Using model type: {model_type}")

if model_type == 'mlp':
    model_files = ['deep_model_0.pkl', 'deep_model_1.pkl']
elif model_type == 'logistic':
    model_files = ['comlex_model_0.pkl', 'comlex_model_1.pkl']
else:
    raise ValueError("Invalid MODEL_TYPE environment variable. Should be 'mlp' or 'logistic'.")

wnl = WordNetLemmatizer()
tknzr = TweetTokenizer(preserve_case=True, reduce_len=True)
english_words = set(words.words())
trans_dict = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”", u"'''\"\"")])

def threshold(label, model):
    print(label, model)
    test_path = os.path.join(data_path, f"test_{model}_pred.csv")
    test_df = pd.read_csv(test_path)
    label_true = [int(eval(l)[labels[label]]) for l in test_df["labels"].values]
    label_score = [float(eval(l)[labels[label]]) for l in test_df["predictions"].values]
    label_score.sort(reverse=True)
    chosen_threshold = float(label_score[sum(label_true)])
    assert sum([l > chosen_threshold for l in label_score]) == sum(label_true)
    print("Chosen threshold:\t", round(chosen_threshold, 3))
    
    label_predict = [1 if l > chosen_threshold else 0 for l in label_score]
    fp = [t == 0 and p == 1 for t, p in zip(label_true, label_predict)]
    fn = [t == 1 and p == 0 for t, p in zip(label_true, label_predict)]
    contingency = [
        [sum(fp), len(fp) - sum(fp)],
        [sum(fn), len(fn) - sum(fn)],
    ]
    chi2, p_value, dof, ex = chi2_contingency(contingency, correction=False)
    consistency = bool(p_value > 0.01 / 6)
    print("Consistent?\t\t", consistency, float(p_value))
    
    return {"threshold": round(chosen_threshold, 3), "consistent": consistency, "p_value": round(float(p_value), 3)}
if __name__ == "__main__":
    sys_path = sys.path[0].split("src")[0]
    data_path = os.path.join(sys_path, "data")
    test_path = os.path.join(data_path, "test.csv")
    labels = {"belief": 0, "disbelief": 1}
    models = [f"{model_type}+LR"]
    thresholds = {label: {model.lower(): 0 for model in models} for label in labels}

    threshold_results = {}
    for label in labels:
        for model in models:
            threshold_data = threshold(label, model.lower())
            threshold_results[f"{label}_{model.lower()}"] = threshold_data
    thresholds_path = os.path.join(sys_path, "results", "thresholds_and_consistency.json")
    with open(thresholds_path, 'w') as f:
        json.dump(threshold_results, f)
    print(f"Thresholds and consistency data saved to {thresholds_path}")
    