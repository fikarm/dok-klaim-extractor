import matplotlib as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

DATASET_PATH = Path("dataset", "20251030124033.csv.preprocess.csv")


def load_dataset() -> pd.DataFrame:
    # load dataset
    dataset = pd.read_csv(DATASET_PATH)
    # c = dataset["label"].value_counts()
    # print(c)

    return dataset


def training_linear(x_train, x_test, y_train, y_test):
    # latih model
    model = SVC(kernel="linear")
    model.fit(x_train, y_train)

    # prediksi label untuk data pengujian
    y_pred = model.predict(x_test)

    # evaluasi model
    print("akurasi:", accuracy_score(y_test, y_pred))
    print("ringakasan:\n", classification_report(y_test, y_pred, labels=y_pred))
    # print("ringakasan:\n", classification_report(y_test, y_pred, labels=np.unique(y_pred)))


def training_rbf(x_train, x_test, y_train, y_test):
    # latih model
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(x_train, y_train)

    # prediksi label untuk data pengujian
    y_pred = model.predict(x_test)

    # evaluasi model
    print("akurasi:", accuracy_score(y_test, y_pred))
    print("ringakasan:\n", classification_report(y_test, y_pred, labels=y_pred))
    # print("ringakasan:\n", classification_report(y_test, y_pred, labels=np.unique(y_pred)))


def visualisasi():
    pass


dataset = load_dataset()
dataset = dataset.drop(index=186)  # hapus single row
# print(dataset.tail())

# ekstraksi fitur menggunakan frekuensi kata dibanding dokumen
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(dataset["content"])
y = dataset["label"]
print(x.shape)

# pecah dataset dengan komposisi 80% untuk pelatihan dan 20% untuk pengujian
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# print(x_test[0])
training_linear(x_train, x_test, y_train, y_test)

