import re
import nltk
import string
import pandas as pd

from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


DATASET_PATH = Path("dataset", "20251030124033.csv")


def download_nltk_stopwords():
    """
    jalankan sekali saja di awal
    """
    nltk.download("stopwords")


def load_dataset() -> pd.DataFrame:
    # load dataset
    dataset = pd.read_csv(DATASET_PATH)
    # c = dataset["label"].value_counts()
    # print(c)

    return dataset


def encode_label(labels: pd.Series):
    le = LabelEncoder()
    return le.fit_transform(labels)


def preproses_teks(text: str) -> str:
    print()
    print("Preproses:", text[:20])

    # lowercasing
    text = text.lower()
    print("- lowercased")

    # hapus tanda baca
    text = re.sub(f"[{string.punctuation}]", " ", text)
    print("- punctuation cleaned")

    # hapus angka
    text = re.sub(r"\d+", " ", text)
    print("- number cleaned")

    # hilangkan multi spasi
    text = re.sub(r"\s+", " ", text)
    print("- whitespace collapsed")

    # pecah kalimat menjadi kata-kata (token)
    tokens = text.split()
    print("- tokenization done")

    # buang stopwords (kata sambung) bahasa indonesia
    stopwords_list = set(nltk.corpus.stopwords.words("indonesian"))
    tokens = [word for word in tokens if word not in stopwords_list]
    print("- id stopwords")

    # buang stopwords (kata sambung) bahasa inggris
    stopwords_list = set(nltk.corpus.stopwords.words("english"))
    tokens = [word for word in tokens if word not in stopwords_list]
    print("- en stopwords")

    # NOTE: perlu proses agak lama
    # # stemming (ubah menjadi kata dasar) untuk kata berbahasa indonesia
    # stemmer = StemmerFactory().create_stemmer()
    # tokens = [stemmer.stem(word) for word in tokens]
    # print("- id stemming")

    # NOTE: perlu proses agak lama
    # # stemming kata berbahasa inggris
    # stemmer = nltk.stem.PorterStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]
    # print("- en stemming")

    # gabung token mejadi teks kembali
    return " ".join(tokens)


dataset = load_dataset()

# encode label
labels = encode_label(dataset["label"])
# print(labels[:3])

# preproses konten
contents = [preproses_teks(content) for content in dataset["content"]]
print(contents[:3])

df = pd.DataFrame({"content": contents, "label": labels})
df.to_csv(f"{DATASET_PATH}.preprocess.csv", index=False)
