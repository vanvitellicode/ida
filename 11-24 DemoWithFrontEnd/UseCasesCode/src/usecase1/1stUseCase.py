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

# Inizializza componenti NLTK
wnl = WordNetLemmatizer()
tknzr = TweetTokenizer(preserve_case=True, reduce_len=True)
english_words = set(words.words())
trans_dict = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”", u"'''\"\"")])

# Funzioni per processamento ed evaluation
def threshold(label, model):
    print(label, model)
    test_path = os.path.join(data_path, f"test_{model}_pred.csv")
    test_df = pd.read_csv(test_path)
    label_true = [eval(l)[labels[label]] for l in test_df["labels"].values]
    label_score = [eval(l)[labels[label]] for l in test_df["predictions"].values]
    label_score.sort(reverse=True)
    threshold = label_score[sum(label_true)]
    assert sum([l > threshold for l in label_score]) == sum(label_true)
    print("Chosen threshold:\t", round(threshold, 3))
    label_predict = [1 if l > threshold else 0 for l in label_score]
    fp = [t == 0 and p == 1 for t, p in zip(label_true, label_predict)]
    fn = [t == 1 and p == 0 for t, p in zip(label_true, label_predict)]
    contingency = [
        [sum(fp), len(fp) - sum(fp)],
        [sum(fn), len(fn) - sum(fn)],
    ]
    chi2, p_value, dof, ex = chi2_contingency(contingency, correction=False)
    print("Consistent?\t\t", p_value > 0.01 / 6, p_value)
    return threshold

def evaluate(model):
    path = os.path.join(data_path, f"test_{model}_pred.csv")
    df = pd.read_csv(path)

    belief = {}
    belief_true = [eval(l)[0] for l in df["labels"].values]
    belief_score = [eval(l)[0] for l in df["predictions"].values]
    belief_predict = [1 if l > thresholds["belief"][model] else 0 for l in belief_score]
    print("\nBelief")
    metrics_belief = {}
    for average in ["binary", "macro", "micro"]:
        f1 = round(f1_score(belief_true, belief_predict, average=average), 3)
        print(model, average + " F1:", f1)
        metrics_belief[f"f1_{average}"] = f1
    belief["precision"], belief["recall"], belief["threshold"] = precision_recall_curve(belief_true, belief_score)
    belief["precision"] = [0] + list(belief["precision"]) + [1]
    belief["recall"] = [1] + list(belief["recall"]) + [0]

    disbelief = {}
    disbelief_true = [eval(l)[1] for l in df["labels"].values]
    disbelief_score = [eval(l)[1] for l in df["predictions"].values]
    disbelief_predict = [1 if l > thresholds["disbelief"][model] else 0 for l in disbelief_score]
    print("\nDisbelief")
    metrics_disbelief = {}
    for average in ["binary", "macro", "micro"]:
        f1 = round(f1_score(disbelief_true, disbelief_predict, average=average), 3)
        print(model, average + " F1:", f1)
        metrics_disbelief[f"f1_{average}"] = f1
    disbelief["precision"], disbelief["recall"], disbelief["threshold"] = precision_recall_curve(disbelief_true, disbelief_score)
    disbelief["precision"] = [0] + list(disbelief["precision"]) + [1]
    disbelief["recall"] = [1] + list(disbelief["recall"]) + [0]

    return belief, disbelief, metrics_belief, metrics_disbelief

def get_label(row):
    if row["DOES NOT"] == 1:
        return [0, 1]  
    elif row["DOES"] == 1:
        return [1, 0] 
    else:
        return [0, 0] 

def pretoken_parse(t):
    if not t:
        return t
    elif "http" in t.lower():
        return "URLREF"
    else:
        return t

def posttoken_parse(t):
    if not t:
        return t
    elif t[0] == "@":
        return "USERREF"
    elif t.replace(".", "").replace(",", "").isdigit():
        return "NUMREF"
    else:
        return t

def tokenize(p):
    tokens = " ".join([pretoken_parse(token) for token in str(p).split(" ")])
    tokens = tknzr.tokenize(tokens)  # lower and tokenize
    num_english_words = [token in english_words for token in tokens]
    if sum(num_english_words) < 0.05 * len(tokens):  # At least 5% English words.
        return np.nan
    tokens = " ".join([posttoken_parse(token) for token in tokens])
    tokens = tokens.translate(trans_dict).strip()
    tokens = tokens.replace("\n", " ")
    if tokens[:3] == "RT ":
        tokens = " ".join(tokens[3:].split(":")[1:])
    tokens = html.unescape(tokens)
    tokens = re.sub("\s+", " ", tokens).strip()
    return tokens

# Funzioni aggiuntive per ComLex
tknzr = TweetTokenizer(preserve_case=False, reduce_len=True)

def get_linguistic(t, ts):
    return len(set(t.split(" ")).intersection(ts))

def lemmatize(t):
    tokens = t.lower().split()
    tokens = [wnl.lemmatize(token, "n") for token in tokens]  # lemmatization noun
    tokens = [wnl.lemmatize(token, "v") for token in tokens]  # lemmatization verb
    return " ".join(tokens)

def label_comlex(df):
    for name, info, tokens in lex.values:
        ts = tokens.split("|")
        col_name = ":::".join([str(name), str(info), "|".join(ts[:5])])
        df[col_name] = df["text"].apply(lambda t: get_linguistic(t, ts))
    return df

def get_chance(l):
    return [random.random(), random.random()]

# Esecuzione principale
if __name__ == "__main__":
    # Percorsi
    sys_path = sys.path[0].split("src")[0]
    corpus_path = os.path.join(sys_path, "resources", "corpus_token.pkl")
    data_path = os.path.join(sys_path, "data")
    raw_path = os.path.join(data_path, "annotations")
    test_path = os.path.join(data_path, "test.csv")
    predict_path = os.path.join(data_path, "predict.csv")
    print(raw_path)

    # Pulizia dei dati annotati
    dfs = []
    for _, _, file_names in os.walk(raw_path):
        for file_name in file_names:
            file_path = os.path.join(raw_path, file_name)
            df = pd.read_excel(file_path, skiprows=1)
            df["text"] = df["Unnamed: 2"]
            df["labels"] = df[["DOES NOT", "DOES"]].apply(get_label, axis=1)
            df = df.dropna(subset={"text"})
            df["text"] = df["text"].apply(tokenize)
            df = df.dropna(subset={"text"})
            df["group"] = file_name.split(", ")[2].split(" tweet")[0]
            dfs.append(df[["group", "text", "labels"]])
    df = pd.concat(dfs).reset_index()

    # Usa l'intero dataset come set di test
    test = df.copy()

    # Salva i dati
    df[["text", "labels"]].dropna().to_csv(test_path, index=False)
    df = pd.read_csv(test_path)
    df = df.dropna()
    df.to_csv(test_path, index=False)
    print("Done preparing test set.")

    # Label ComLex
    lex_path = os.path.join(sys_path, "resources", "ComLex.csv")
    lex = pd.read_csv(lex_path)

    # Labels ComLex
    for path in [test_path]:
        df = pd.read_csv(path)
        df["lemmas"] = df["text"].apply(lemmatize)
        df = label_comlex(df)
        df.to_csv(path.replace(".csv", "_comlex.csv"), index=False)

    # Predizioni casuali
    test_chance_path = os.path.join(data_path, "test_chance_pred.csv")
    chance_df = pd.read_csv(test_path)
    chance_df["labels"] = chance_df["labels"].apply(lambda l: eval(l))
    chance_df["predictions"] = chance_df["labels"].apply(get_chance)
    chance_df.to_csv(test_chance_path, index=False)

    # Predizioni ComLex
    test_comlex_path = os.path.join(data_path, f"test_{model_type}+lr_pred.csv")
    test_path = os.path.join(data_path, "test_comlex.csv")
    test_df = pd.read_csv(test_path)

    pred_Y = {"test": []}
    for label in [0, 1]:
        test_X = test_df[test_df.columns[3:]].values
        print("Feature vector example:", test_X[0])

        # Carica i modelli basati sulla variabile d'ambiente
        with open(os.path.join(data_path, model_files[label]), "rb") as f:
            complex_model = pickle.load(f)

        pred_Y_label = complex_model.predict_proba(test_X)
        pred_Y["test"].append([prob[1] for prob in pred_Y_label])

        print(f"Predictions for label {label}:", pred_Y_label)

    # Combina le predizioni
    predictions = [[pred1, pred2] for pred1, pred2 in zip(pred_Y["test"][0], pred_Y["test"][1])]

    test_df["predictions"] = predictions
    test_df[["text", "labels", "predictions"]].to_csv(test_comlex_path, index=False)

    # Valutazione
    labels = {"belief": 0, "disbelief": 1}
    models = [f"{model_type}+LR"]
    thresholds = {label: {model.lower(): 0 for model in models} for label in labels}

    # Trova le soglie
    for label in labels:
        for model in models:
            thresholds[label][model.lower()] = threshold(label, model.lower())

    # Valuta i modelli e raccogli le metriche
    prc = {model: {} for model in models}
    metrics_results = {}
    for model in models:
        belief, disbelief, metrics_belief, metrics_disbelief = evaluate(model.lower())
        prc[model]["belief"] = belief
        prc[model]["disbelief"] = disbelief
        metrics_results['belief'] = metrics_belief
        metrics_results['disbelief'] = metrics_disbelief

    # Salva le metriche in un file JSON
    metrics_path = os.path.join(sys_path, "results", "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_results, f)
    print(f"Metrics saved to {metrics_path}")

    # Parametri per il plotting (manteniamo questa parte come è)
    plot_args = {
        "belief": {
            "Chance": {"c": "k", "ls": "-"},
            "LIWC+LR": {"c": "#AAAAAA", "ls": ":"},
            f"{model_type}+LR": {"c": "#AAAAAA", "ls": "-"},
            "XLNet": {"c": "seagreen", "ls": "--"},
            "BERT": {"c": "seagreen", "ls": ":"},
            "RoBERTa": {"c": "seagreen", "ls": "-"},
        },
        "disbelief": {
            "Chance": {"c": "k", "ls": "-"},
            "LIWC+LR": {"c": "#AAAAAA", "ls": ":"},
            f"{model_type}+LR": {"c": "#AAAAAA", "ls": "-"},
            "XLNet": {"c": "indianred", "ls": "--"},
            "BERT": {"c": "indianred", "ls": ":"},
            "RoBERTa": {"c": "indianred", "ls": "-"},
        },
    }

    for split in ["test"]:
        for label in ["belief", "disbelief"]:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

            # Plot F1 isoheights
            f_scores = np.linspace(0.2, 0.8, num=4)
            x_locs = [0.16, 0.35, 0.54, 0.73]
            y_locs = [0.295, 0.495, 0.695, 0.895]
            for x_loc, y_loc, f_score in zip(x_locs, y_locs, f_scores):
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2, linewidth=1)
                ax.annotate(
                    f"F$_1$={f_score:0.1f}", xy=(x_loc, y_loc), alpha=0.3, fontproperties=font
                )

            # Plot precision-recall curve del modello
            for model in models:
                ax.plot(
                    prc[model][label]["recall"],
                    prc[model][label]["precision"],
                    label=model,
                    alpha=0.8,
                    color=plot_args[label][model]["c"],
                    linestyle=plot_args[label][model]["ls"],
                )

            # Configura il plot
            ax.set_xlim([0, 1.005])
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_xlabel("Recall", fontproperties=font2)
            ax.set_ylim([0, 1.005])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_ylabel("Precision", fontproperties=font2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            leg = ax.legend(
                bbox_to_anchor=(0.5, -0.18, 0, 0), loc="upper center", ncol=2, borderaxespad=0.0
            )
            leg.get_frame().set_alpha(0)

            # Salva la figura
            fig_path = os.path.join(sys_path, "results", f"{split}_{label}_pr_re_curve.pdf")
            plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)

    print("Processing complete.")