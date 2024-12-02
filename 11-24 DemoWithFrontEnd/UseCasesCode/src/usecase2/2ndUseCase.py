import random
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier  # Added MLPClassifier import
import pickle
import os, sys, re, html
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
import nltk
nltk.download('wordnet')
nltk.download('words')
from nltk.corpus import words

wnl = WordNetLemmatizer()
tknzr = TweetTokenizer(preserve_case=True, reduce_len=True)
english_words = set(words.words())
trans_dict = dict([(ord(x), ord(y)) for x,y in zip(u"‘’´“”",  u"'''\"\"")])

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
    elif t.replace(".","").replace(",","").isdigit():
        return "NUMREF"
    else:
        return t

def tokenize(p):
    tokens = " ".join([pretoken_parse(token) for token in str(p).split(" ")])
    tokens = tknzr.tokenize(tokens)
    num_english_words = [token in english_words for token in tokens]
    if sum(num_english_words) < 0.05 * len(tokens):  # At least 5% English words.
        return np.nan
    tokens = " ".join([posttoken_parse(token) for token in tokens])
    tokens = tokens.translate(trans_dict).strip()
    tokens = tokens.replace("\n", " ")
    if tokens[:3] == "RT ":
        tokens = " ".join(tokens[3:].split(":")[1:])
    tokens = html.unescape(tokens)
    tokens = re.sub(r"\s+", " ", tokens).strip()
    return tokens

tknzr = TweetTokenizer(preserve_case=False, reduce_len=True)

def get_linguistic(t, ts):
    return len(set(t.split(" ")).intersection(ts))

def lemmatize(t):
    tokens = t.lower().split()
    tokens = [wnl.lemmatize(token, "n") for token in tokens]  # Lemmatization noun
    tokens = [wnl.lemmatize(token, "v") for token in tokens]  # Lemmatization verb
    return " ".join(tokens)

def label_comlex(df):
    for name, info, tokens in lex.values:
        ts = tokens.split("|")
        col_name = ":::".join([str(name), str(info), "|".join(ts[:5])])
        df[col_name] = df["text"].apply(lambda t: get_linguistic(t, ts))
    return df

def get_chance(l):
    return [random.random(), random.random()]

if __name__ == "__main__":
    # Setup paths
    sys_path = sys.path[0].split("src")[0]
    data_path = os.path.join(sys_path, "data")
    raw_path = os.path.join(data_path, "annotations")
    train_path = os.path.join(data_path, "train.csv")
    lex_path = os.path.join(sys_path, "resources", "ComLex.csv")

    print(raw_path)
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
    df = pd.concat(dfs).reset_index(drop=True)
    df[["text", "labels"]].dropna().to_csv(train_path, index=False)
    df = pd.read_csv(train_path)
    df = df.dropna()
    df.to_csv(train_path, index=False)
    print("Done train and test sets.")

    # Load lexicon
    lex = pd.read_csv(lex_path)

    # Process training data
    for path in [train_path]:
        df = pd.read_csv(path)
        df["lemmas"] = df["text"].apply(lemmatize)
        df = label_comlex(df)
        df.to_csv(path.replace(".csv", "_comlex.csv"), index=False)

    print(sys_path)

    # Generate chance predictions
    train_chance_path = os.path.join(data_path, "train_chance_pred.csv")
    chance_df = pd.read_csv(train_path)
    chance_df["labels"] = chance_df["labels"].apply(lambda l: eval(l))
    chance_df["predictions"] = chance_df["labels"].apply(get_chance)
    chance_df.to_csv(train_chance_path, index=False)

    # Train model based on environment variable
    train_comlex_path = os.path.join(data_path, "train_comlex+lr_pred.csv")
    train_path = os.path.join(data_path, "train_comlex.csv")
    train_df = pd.read_csv(train_path)
    pred_Y = {"train": []}

    # Get model type from environment variable
    model_type = os.environ.get('MODEL_TYPE', 'logistic')  # Default to 'logistic' if not set

    for label in [0, 1]:
        train_Y = train_df["labels"].apply(lambda l: eval(l)[label]).values
        print(eval(train_df["labels"].iloc[0]), eval(train_df["labels"].iloc[0])[0], eval(train_df["labels"].iloc[0])[1])
        print(eval(train_df["labels"].iloc[1]), eval(train_df["labels"].iloc[1])[0], eval(train_df["labels"].iloc[1])[1])
        print(train_df["labels"].head())
        print(train_Y[:2])
        train_X = train_df.iloc[:, 3:].values

        # Initialize model based on environment variable
        if model_type == 'logistic':
            model = LogisticRegression(random_state=0, solver="lbfgs", max_iter=10000)
            model_name = f"comlex_model_{label}.pkl"
        elif model_type == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=(100,), random_state=0, max_iter=10000)
            model_name = f"deep_model_{label}.pkl"
        else:
            raise ValueError(f"Unsupported MODEL_TYPE environment variable value: {model_type}")

        model.fit(train_X, train_Y)
        pickle.dump(model, open(os.path.join(data_path, model_name), "wb"))
        pred_Y["train"].append([prob[1] for prob in model.predict_proba(train_X)])

    predictions = [[pred1, pred2] for pred1, pred2 in zip(pred_Y["train"][0], pred_Y["train"][1])]
    train_df["predictions"] = predictions
    train_df[["text", "labels", "predictions"]].to_csv(train_comlex_path, index=False)
    print("Done train set.")