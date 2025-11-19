import os
import gdown
import streamlit as st
import pickle
import re

# Імпорти для класифікаторів
from sklearn.pipeline import Pipeline
from classifiers.BertBaseClassifier import BertBaseClassifier
from classifiers.BertBaseMultiClassifier import BertBaseMultiClassifier
from classifiers.KerasModelClassifier import CNNGLTRClassifier
from classifiers.SvmTfidfBaseClassifier import SvmTfidfBaseClassifierV2
import pandas as pd
import numpy as np
from input_preprocess.GLTRTransformer import LM, GLTRTransformer
from models.ensemble.Ensemble import Ensemble
from lime.lime_text import LimeTextExplainer
from annotated_text import annotation
from nltk.tokenize import sent_tokenize
import re

# --- КОНСТАНТИ GOOGLE DRIVE ID та ШЛЯХИ ---

# ВАШІ ID:
SVM_FILE_ID = "1_O8mtsgJipuCgqrW1yBoJBUUEYCiUXsx"
SVM_VECTORIZER_FILE_ID = "1HBpIvydoh6slZKwrX4o9hsinGPydwkIi" # Судячи з шляху, це файл "tfidf_vectorizer_90000_features.pkl"
BERT_FILE_ID = "1D8wp3sOVV9Ri5BUG26IGVSZoSZlvjobD"
BERT_MULTICLASS_FILE_ID = "1GhTr-2ghquSTWdha96s7JJWegx2yoo2t"
# CNN (model_autokeras_gltr_trials_8) - ця папка завантажується інакше,
# або вона має бути на GitHub, якщо вона не перевищує ліміт.

# Шляхи, які будуть використовуватися в коді для локального доступу
SVM_MODEL_PATH = "models/svm_linear_model_90000_features_probability.pkl"
SVM_VECTORIZER_PATH = "models/tfidf_vectorizer_90000_features.pkl"
BERT_MODEL_PATH = "models/model_bertbase_updated.pt"
BERT_MULTICLASS_PATH = "models/model_multiclass.pt"
CNN_MODEL_PATH = "models/model_autokeras_gltr_trials_8" # Це папка, її поки залишимо на GitHub, якщо вона не була великою

# --- МЕХАНІЗМ ЗАВАНТАЖЕННЯ З GOOGLE DRIVE ---

@st.cache_resource
def get_model_from_drive(file_id, output_path, is_pickle=True):
    """Завантажує файл з Google Drive, кешує його та повертає шлях або десеріалізований об'єкт."""
    
    # 1. Створення директорії
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 2. Завантаження, якщо файл відсутній
    if not os.path.exists(output_path):
        st.info(f"Downloading model {os.path.basename(output_path)}...")
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            st.success("Model file downloaded.")
        except Exception as e:
            st.error(f"Failed to download model from Google Drive: {e}")
            return None # Повертаємо None у разі помилки

    # 3. Повернення об'єкта (якщо це pickle) або шляху
    if is_pickle:
        try:
            with open(output_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load pickled object from disk: {e}")
            return None
    else:
        # Для .pt або папок, повертаємо лише шлях, оскільки ініціалізація моделі (torch.load)
        # відбувається безпосередньо у класах класифікаторів.
        return output_path

# --- ІНТЕЛЕКТУАЛЬНЕ ЗАВАНТАЖЕННЯ ТА ІНІЦІАЛІЗАЦІЯ МОДЕЛЕЙ ---

@st.cache_resource
def initialize_models():
    """
    Завантажує всі необхідні компоненти (моделі та векторизатори) та ініціалізує класифікатори.
    Ця функція викликається лише один раз завдяки @st.cache_resource.
    """
    st.write("Initializing models...")
    
    # Завантаження SVM моделі (за допомогою нової логіки)
    # Зверніть увагу: SvmTfidfBaseClassifierV2 очікує шляхи, а не об'єкти
    # Оскільки SVM V2, ймовірно, завантажує векторизатор окремо, нам потрібен шлях.
    
    # Завантажуємо окремо модель SVM
    svm_model_obj = get_model_from_drive(SVM_FILE_ID, SVM_MODEL_PATH, is_pickle=True)
    # Завантажуємо окремо TFIDF векторизатор (припускаємо, що він теж великий і був винесений)
    tfidf_vectorizer_obj = get_model_from_drive(SVM_VECTORIZER_FILE_ID, SVM_VECTORIZER_PATH, is_pickle=True)
    
    # Моделі BERT та Multiclass BERT очікують лише шлях до файлу .pt
    bert_path = get_model_from_drive(BERT_FILE_ID, BERT_MODEL_PATH, is_pickle=False)
    bert_multiclass_path = get_model_from_drive(BERT_MULTICLASS_FILE_ID, BERT_MULTICLASS_PATH, is_pickle=False)
    
    # Перевіряємо, чи все завантажилося
    if any(obj is None for obj in [svm_model_obj, tfidf_vectorizer_obj, bert_path, bert_multiclass_path]):
        st.error("One or more critical models failed to load. Please check Google Drive IDs and access.")
        return None, None, None, None # Повертаємо None, щоб запобігти збоям

    # Ініціалізуємо класифікатори, передаючи ШЛЯХИ
    # Для SvmTfidfBaseClassifierV2, ми тимчасово завантажимо об'єкти, 
    # а потім створимо класифікатори, використовуючи ШЛЯХИ, щоб відповідати старому коду.
    # Оскільки ми вже завантажили файли, ми просто передаємо шляхи:
    svm = SvmTfidfBaseClassifierV2(SVM_MODEL_PATH, SVM_VECTORIZER_PATH)
    bert = BertBaseClassifier(BERT_MODEL_PATH)
    cnn = CNNGLTRClassifier(CNN_MODEL_PATH) # CNN_MODEL_PATH припускаємо, що залишається на GitHub
    bert_multiclass = BertBaseMultiClassifier(BERT_MULTICLASS_PATH)
    
    return svm, bert, cnn, bert_multiclass


# Ініціалізація моделей (відбудеться при першому запуску кнопки Check, якщо ви внесли зміни в App/Lite.py)
svm, bert, cnn, bert_multiclass = initialize_models()

# Якщо ініціалізація провалилася, зупиняємо
if svm is None:
    st.stop()


# Інстанціювання Ensemble та Pipeline (залишається як було, але використовує завантажені моделі)
pipeline = Pipeline(steps=[('GLTR', GLTRTransformer())])
models = [bert, cnn, svm]
ensemble = Ensemble(models, ["BERT", "CNN", "SVM"])
weights = np.array([0.25, 0.25, 0.5])


# --- ВАШ ІНШИЙ КОД БЕЗ ЗМІН ---

def chunk_into_even_paragraphs(text, max_chunk_size):
    B = len
    I = max_chunk_size
    J = sent_tokenize(text)
    P = max([B(A)for A in J])
    E, C, F = [], [], 0
    for G in J:
        if F+B(G)+B(C) > I:
            E.append(C)
            C, F = [], 0
        C.append(G)
        F += B(G)+B(C)
    E.append(C)
    D = [[B(A)for A in A]for A in E]
    H = True
    while H:
        H = False
        for A in range(B(D)-1, 0, -1):
            K, C = sum(D[A])+B(D[A]), sum(D[A-1])+B(D[A-1])
            L = abs(K-C)
            if (M := K+D[A-1][-1]+1) < I:
                N = C-D[A-1][-1]-1
                O = abs(M-N)
                if O < L:
                    E[A].insert(0, E[A-1].pop())
                    D[A].insert(0, D[A-1].pop())
                    H = True
    return list(map(lambda p: ' '.join(p), E))


def split_into_paragraphs(text, max_char_limit):
    # split the text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)

    output = []
    for paragraph in paragraphs:
        if len(paragraph) > max_char_limit:
            # split the paragraph into sentences
            chunks = chunk_into_even_paragraphs(paragraph, max_char_limit)
            output.extend(chunks)
        else:
            output.append(paragraph)

    return output


def check_if_ai(text: str, threshold: int) -> tuple:
    splitted_text = split_into_paragraphs(text, 2000)
    predictions = []
    individual_scores = {}
    for i, text in enumerate(splitted_text):
        text_lst = [text]
        sample_df = pd.DataFrame(text_lst, columns=['response'])
        processed_input_df = pipeline.fit_transform(sample_df)
        pred, output_dict = ensemble.predict(
            processed_input_df, weights, threshold)
        scores = {key: val[0] for key, val in output_dict.items()}
        predictions.append(pred)
        individual_scores[i+1] = scores

    # generate the average individual score
    average_scores = {}
    for key, value in individual_scores.items():
        for k, v in value.items():
            if k in average_scores:
                average_scores[k] += v
            else:
                average_scores[k] = v
    for key, value in average_scores.items():
        average_scores[key] = value / len(individual_scores)
    individual_scores["Average"] = average_scores
    return predictions, individual_scores, splitted_text


def check_if_ai_short_text(text: str, threshold: int) -> tuple:
    text_lst = [text]
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    processed_input_df = pipeline.fit_transform(sample_df)
    pred, output_dict = ensemble.predict(
        processed_input_df, weights, threshold)
    scores = {key: val[0] for key, val in output_dict.items()}
    return pred, scores


def check_ai_percentage(predictions: list):
    # find the mean of the predictions
    ai_text_count = predictions.count("AI")
    percentage_AI = ai_text_count / len(predictions)
    return percentage_AI


def check_if_paraphrased(text: str) -> bool:
    splitted_text = split_into_paragraphs(text, 2000)
    predictions = []
    for text in splitted_text:
        text_lst = [text]
        sample_df = pd.DataFrame(text_lst, columns=['response'])
        prediction = bert_multiclass.predict(sample_df)
        predictions.append(prediction[0].detach().numpy())
    return predictions


def check_if_paraphrased_short_text(text: str):
    text_lst = [text]
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    prediction = bert_multiclass.predict(sample_df)
    return prediction[0].detach().numpy()


def check_if_paraphrased_percentage(predictions: list, threshold: int):
    # count the number of paraphrased text
    paraphrased_text_count = 0
    for prediction in predictions:
        if prediction[2] > threshold:
            paraphrased_text_count += 1
    percentage_paraphrased = paraphrased_text_count / len(predictions)

    return percentage_paraphrased


def check_if_ai_speed(text: str) -> bool:
    splitted_text = split_into_paragraphs(text, 2000)
    predictions = []
    for text in splitted_text:
        text_lst = [text]
        sample_df = pd.DataFrame(text_lst, columns=['response'])
        prediction = svm.predict(sample_df)
        predictions.append(prediction[0])
    return predictions, splitted_text


def check_if_ai_speed_short_text(text: str) -> bool:
    text_lst = [text]
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    prediction = svm.predict(sample_df)
    return prediction[0]


def check_ai_percentage_speed(predictions: list):
    # find the mean of the predictions
    ai_text_count = predictions.count(1)
    percentage_AI = ai_text_count / len(predictions)
    return percentage_AI


def predict_proba_svm(text_lst: list):
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    pred = svm.predict_proba(sample_df)
    return np.array(pred)


def predict_proba_bert(text_lst: list):
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    pred = bert.predict(sample_df)
    returnable = []
    for i in pred:
        temp = i
        returnable.append(np.array([1-temp, temp]))
    return np.array(returnable)


def predict_proba_cnn(text_lst: list):
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    # Припускаємо, що CNN використовує svm для цього, але це може бути помилкою в оригінальному коді. 
    # Залишаю як було, але зауважте, що тут використовується 'svm' замість 'cnn'.
    pred = svm.predict(sample_df) 
    returnable = []
    for i in pred:
        temp = i
        returnable.append(np.array([1-temp, temp]))
    return np.array(returnable)


def get_explaination(text: str, num_features: int, model: str):
    explainer = LimeTextExplainer(class_names=["Human", "AI"], bow=False)
    if model == "SVM":
        exp = explainer.explain_instance(
            text, predict_proba_svm, num_features=num_features)
    elif model == "BERT":
        exp = explainer.explain_instance(
            text, predict_proba_bert, num_features=num_features)
    elif model == "CNN":
        exp = explainer.explain_instance(
            text, predict_proba_cnn, num_features=num_features)
    return exp.as_html()


def generate_annotated_text(text: list, labels: list, paraphrase_scores=None, paraphrased_threshold=None):
    if paraphrase_scores is None:
        data = []
        colors = {
            "Human": "#afa",
            "AI": "#fea"
        }
        for i in range(len(text)):
            data.append((text[i], labels[i], colors[labels[i]]))
        return data
    else:
        data = []
        colors = {
            "Human": "#afa",
            "AI": "#fea",
        }
        for i in range(len(text)):
            # only create a border around the text if it is paraphrased AI
            if paraphrase_scores[i][2] > paraphrased_threshold and labels[i] == "AI":
                data.append(annotation(
                    text[i], labels[i], colors[labels[i]], border="2px dashed red"))
            else:
                data.append((text[i], labels[i], colors[labels[i]]))
        return data


def generate_annotated_text_speed(text: list, labels: list):
    data = []
    colors = {
        0: "#afa",
        1: "#fea"
    }
    text_labels = {
        0: "Human",
        1: "AI"
    }
    for i in range(len(text)):
        data.append((text[i], text_labels[labels[i]], colors[labels[i]]))
    return data


def has_cyrillic(text: str):
    return bool(re.search('[\u0400-\u04FF]', text))