import collections as coll
import math
import pickle
import string
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import nltk

nltk.download('cmudict')
nltk.download('stopwords')

style.use("ggplot")
cmuDictionary = None


def slidingWindow(sequence, winSize, step=1):
    if not isinstance(sequence, list):
        raise Exception("**ERROR** sequence must be a list of sentences.")
    if not ((type(winSize) == int) and (type(step) == int)):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    numOfChunks = int(((len(sequence) - winSize) / step) + 1)
    l = []
    for i in range(0, numOfChunks * step, step):
        l.append(" ".join(sequence[i:i + winSize]))
    return l


def syllable_count_Manual(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if len(word) > 0 and word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


def syllable_count(word):
    global cmuDictionary
    d = cmuDictionary
    try:
        syl = [len([y for y in x if y[-1].isdigit()]) for x in d[word.lower()]][0]
    except KeyError:
        syl = syllable_count_Manual(word)
    return syl


def Avg_wordLength(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.lower() not in stop_words]
    if not words:
        return 0
    return np.average([len(word) for word in words])


def Avg_SentLenghtByCh(text):
    tokens = sent_tokenize(text)
    if not tokens:
        return 0
    return np.average([len(token) for token in tokens])


def Avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
    if not tokens:
        return 0
    return np.average([len(token.split()) for token in tokens])


def Avg_Syllable_per_Word(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')).union(set(string.punctuation))
    words = [word for word in tokens if word.lower() not in stop_words]
    if not words:
        return 0
    syllables = [syllable_count(word) for word in words]
    return np.average(syllables)


def CountSpecialCharacter(text):
    special_chars = set("#$%&()*+-/<=>@[\]^_`{|}~\t\n")
    count = sum(1 for char in text if char in special_chars)
    if not text:
        return 0
    return count / len(text)


def CountPuncuation(text):
    punctuation_chars = set(",.'!\";?:;")
    count = sum(1 for char in text if char in punctuation_chars)
    if not text:
        return 0
    return count / len(text)


def CountFunctionalWords(text):
    functional_words = set("""
    a between in nor some upon about both including nothing somebody us
    above but inside of someone used after by into off something via
    all can is on such we although cos it once than what am do its one
    that whatever among down latter onto the when an each less opposite
    their where and either like or our these which any every lots outside
    they while anybody everybody many over this who anyone everyone me own
    those whoever anything everything more past though whom are few most
    per through whose around following much plenty till will as for must
    plus to with at from my regarding toward within be have near same
    towards without because he need several under worth before her neither
    she unless would behind him no should unlike yes below i nobody since
    until you beside if none so up your
    """.split())
    words = RemoveSpecialCHs(text)
    if not words:
        return 0
    count = sum(1 for word in words if word.lower() in functional_words)
    return count / len(words)


def hapaxLegemena(text):
    words = RemoveSpecialCHs(text)
    if not words:
        return 0, 0
    freqs = coll.Counter(words)
    V1 = sum(1 for count in freqs.values() if count == 1)
    N = len(words)
    V = float(len(freqs))
    if V == 0:
        return 0, 0
    R = 100 * math.log(N) / max(1, (1 - (V1 / V)))
    h = V1 / N
    return R, h


def hapaxDisLegemena(text):
    words = RemoveSpecialCHs(text)
    if not words:
        return 0, 0
    freqs = coll.Counter(words)
    count = sum(1 for count in freqs.values() if count == 2)
    h = count / len(words)
    S = count / len(freqs)
    return S, h


def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    if not words:
        return 0
    freqs = coll.Counter(words)
    maximum = float(max(freqs.values()))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])


def typeTokenRatio(text):
    words = word_tokenize(text)
    if not words:
        return 0
    return len(set(words)) / len(words)


def BrunetsMeasureW(text):
    words = RemoveSpecialCHs(text)
    if not words:
        return 0
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    if N == 0:
        return 0
    B = (V - a) / (math.log(N))
    return B


def RemoveSpecialCHs(text):
    tokens = word_tokenize(text)
    st = set(string.punctuation)
    words = [word for word in tokens if word not in st]
    return words


def YulesCharacteristicK(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    if N == 0:
        return 0
    freqs = coll.Counter(words)
    vi = coll.Counter(freqs.values())
    M = sum([(i ** 2) * vi[i] for i in vi])
    K = 10000 * (M - N) / (N ** 2)
    return K


def ShannonEntropy(text):
    words = RemoveSpecialCHs(text)
    if not words:
        return 0
    freqs = coll.Counter(words)
    distribution = np.array(list(freqs.values())) / len(words)
    H = -np.sum(distribution * np.log2(distribution))
    return H


def SimpsonsIndex(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    if N <= 1:
        return 0
    freqs = coll.Counter(words)
    n = sum(count * (count - 1) for count in freqs.values())
    D = 1 - (n / (N * (N - 1)))
    return D


def FleschReadingEase(text, NoOfsentences):
    words = RemoveSpecialCHs(text)
    if not words or NoOfsentences == 0:
        return 0
    scount = sum(syllable_count(word) for word in words)
    l = float(len(words))
    I = 206.835 - 1.015 * (l / NoOfsentences) - 84.6 * (scount / l)
    return I


def FleschCincadeGradeLevel(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    if not words or NoOfSentences == 0:
        return 0
    scount = sum(syllable_count(word) for word in words)
    l = len(words)
    F = 0.39 * (l / NoOfSentences) + 11.8 * (scount / l) - 15.59
    return F


def dale_chall_readability_formula(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    difficult = 0
    adjusted = 0
    NoOfWords = len(words)
    if NoOfWords == 0 or NoOfSentences == 0:
        return 0
    with open('dale-chall.pkl', 'rb') as f:
        familiarWords = pickle.load(f)
    for word in words:
        if word.lower() not in familiarWords:
            difficult += 1
    percent = (difficult / NoOfWords) * 100
    if percent > 5:
        adjusted = 3.6365
    D = 0.1579 * percent + 0.0496 * (NoOfWords / NoOfSentences) + adjusted
    return D


def GunningFoxIndex(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    NoOfWords = float(len(words))
    if NoOfWords == 0 or NoOfSentences == 0:
        return 0
    complexWords = sum(1 for word in words if syllable_count(word) > 2)
    G = 0.4 * ((NoOfWords / NoOfSentences) + 100 * (complexWords / NoOfWords))
    return G


def FeatureExtration(text_entries, winSize, step):
    global cmuDictionary
    cmuDictionary = cmudict.dict()

    chunks = slidingWindow(text_entries, winSize, step)
    vector = []
    for chunk in chunks:
        feature = []
        meanwl = Avg_wordLength(chunk)
        feature.append(meanwl)

        meansl = Avg_SentLenghtByCh(chunk)
        feature.append(meansl)

        mean = Avg_SentLenghtByWord(chunk)
        feature.append(mean)

        meanSyllable = Avg_Syllable_per_Word(chunk)
        feature.append(meanSyllable)

        means = CountSpecialCharacter(chunk)
        feature.append(means)

        p = CountPuncuation(chunk)
        feature.append(p)

        f = CountFunctionalWords(chunk)
        feature.append(f)

        # VOCABULARY RICHNESS FEATURES
        TTratio = typeTokenRatio(chunk)
        feature.append(TTratio)

        HonoreMeasureR, hapax = hapaxLegemena(chunk)
        feature.append(hapax)
        feature.append(HonoreMeasureR)

        SichelesMeasureS, dihapax = hapaxDisLegemena(chunk)
        feature.append(dihapax)
        feature.append(SichelesMeasureS)

        YuleK = YulesCharacteristicK(chunk)
        feature.append(YuleK)

        S = SimpsonsIndex(chunk)
        feature.append(S)

        B = BrunetsMeasureW(chunk)
        feature.append(B)

        Shannon = ShannonEntropy(chunk)
        feature.append(Shannon)

        # READABILITY FEATURES
        FR = FleschReadingEase(chunk, winSize)
        feature.append(FR)

        FC = FleschCincadeGradeLevel(chunk, winSize)
        feature.append(FC)

        D = dale_chall_readability_formula(chunk, winSize)
        feature.append(D)

        G = GunningFoxIndex(chunk, winSize)
        feature.append(G)

        vector.append(feature)

    return vector


def ElbowMethod(data):
    distorsions = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distorsions.append(kmeans.inertia_)

    plt.figure(figsize=(15, 5))
    plt.plot(range(1, 10), distorsions, 'bo-')
    plt.grid(True)
    plt.ylabel("Sum of Squared Errors")
    plt.xlabel("Number of Clusters")
    plt.title('Elbow Curve')
    plt.savefig("ElbowCurve.png")
    plt.show()


def Analysis(vector, K=2):
    arr = np.array(vector)
    sc = StandardScaler()
    x = sc.fit_transform(arr)

    pca = PCA(n_components=2)
    components = pca.fit_transform(x)

    kmeans = KMeans(n_clusters=K)
    kmeans.fit(components)
    print("Labels: ", kmeans.labels_)
    centers = kmeans.cluster_centers_

    labels = kmeans.labels_
    colors = ["r.", "g.", "b.", "y.", "c."]
    colors = colors[:K]

    for i in range(len(components)):
        plt.plot(components[i][0], components[i][1], colors[labels[i]], markersize=10)

    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=150, linewidths=10, zorder=15)
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.title("Styles Clusters")
    plt.savefig("Results.png")
    plt.show()


if __name__ == '__main__':
    with open('frasi.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) 
        text_entries = [row[0] for row in reader]

    vector = FeatureExtration(text_entries, winSize=10, step=10)
    ElbowMethod(np.array(vector))
    Analysis(vector)