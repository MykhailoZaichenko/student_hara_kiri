import streamlit as st
import pandas as pd
import os
import gdown
import plotly.express as px
from annotated_text import annotated_text

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.set_page_config(page_title="Student hara-kiri", page_icon="👨‍🎓")

# --- Налаштування шляхів та ID ---
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

# !!! УВАГА: ЗАМІНІТЬ ЦІ PLACEHOLDER ID НА РЕАЛЬНІ ID ВАШИХ ФАЙЛІВ !!!
MODEL_IDS = {
    # SVM/TF-IDF
    "svm_model": "1_O8mtsgJipuCgqrW1yBoJBUUEYCiUXsx", 
    "tfidf_vectorizer": "1HBpIvydoh6slZKwrX4o9hsinGPydwkIi", 
    # BERT
    "bert_binary": "1D8wp3sOVV9Ri5BUG26IGVSZoSZlvjobD", 
    "bert_multiclass": "1GhTr-2ghquSTWdha96s7JJWegx2yoo2t", 
    # CNN/GLTR (SavedModel components)
    "cnn_saved_model_pb": "1e8ApGwBSC985I0eTjr_jhWMhA1amtEJi",
    "cnn_keras_metadata_pb": "19VuA-EkD-i7h-PoaoELa2lgXMdQGhQ_x",
    "cnn_variables_index": "1ExpVHEL2yan-RxsABYCEVltPwDNHiiH6", 
    "cnn_variables_data_00000": "1Pq9meCh5Q0K1HLqLRLMDHUCkDCPjmjLI",
}

# Шляхи до файлів
PATHS = {
    "svm_model": os.path.join(MODELS_DIR, "svm_linear_model_90000_features_probability.pkl"),
    "tfidf_vectorizer": os.path.join(MODELS_DIR, "tfidf_vectorizer_90000_features.pkl"),
    "bert_binary": os.path.join(MODELS_DIR, "model_bertbase_updated.pt"),
    "bert_multiclass": os.path.join(MODELS_DIR, "model_multiclass.pt"),
    
    # Keras SavedModel вимагає директорії. Ми створюємо потрібну структуру.
    "cnn_model_dir": os.path.join(MODELS_DIR, "model_autokeras_gltr_trials_8"),
    "cnn_variables_dir": os.path.join(MODELS_DIR, "model_autokeras_gltr_trials_8", "variables"),

    "cnn_saved_model_pb": os.path.join(MODELS_DIR, "model_autokeras_gltr_trials_8", "saved_model.pb"),
    "cnn_keras_metadata_pb": os.path.join(MODELS_DIR, "model_autokeras_gltr_trials_8", "keras_metadata.pb"),
    "cnn_variables_index": os.path.join(MODELS_DIR, "model_autokeras_gltr_trials_8", "variables", "variables.index"),
    "cnn_variables_data_00000": os.path.join(MODELS_DIR, "model_autokeras_gltr_trials_8", "variables", "variables.data-00000-of-00001"), 
    # Примітка: Всі інші файли variables.data-***** потрібно завантажити аналогічно!
    # Якщо у вас є більше .data файлів, додайте їх ID в MODEL_IDS та PATHS і цикл завантаження.
}

# Створення необхідних директорій для Keras SavedModel
os.makedirs(PATHS["cnn_variables_dir"], exist_ok=True)


@st.cache_resource(show_spinner=False)
def download_all_files(model_ids, paths):
    """Завантажує всі файли моделей з Google Drive, якщо вони відсутні."""
    
    download_statuses = {}
    
    for key, file_path in paths.items():
        if key.startswith("cnn_") or key.startswith("bert") or key.startswith("svm") or key.startswith("tfidf"):
            
            # Пропускаємо, якщо це директорія
            if key.endswith("_dir"):
                continue

            # Перевіряємо, чи існує файл
            if os.path.exists(file_path):
                download_statuses[key] = f"✅ {os.path.basename(file_path)} already exists."
                continue
            
            # Перевіряємо, чи встановлений ID
            if key not in model_ids or "ВАШ_ID" in model_ids[key]:
                download_statuses[key] = f"⚠️ {os.path.basename(file_path)} ID is missing/default."
                continue

            # Завантаження файлу
            file_id = model_ids[key]
            try:
                gdown.download(f'https://drive.google.com/uc?id={file_id}', file_path, quiet=True)
                download_statuses[key] = f"✅ {os.path.basename(file_path)} downloaded successfully."
            except Exception as e:
                download_statuses[key] = f"❌ ERROR downloading {os.path.basename(file_path)}: {e}. Check ID and permissions."

    return download_statuses

# --- ФАЗА 1: ЗАВАНТАЖЕННЯ ФАЙЛІВ ПРИ СТАРТІ ЗАСТОСУНКУ ---
with st.spinner("Перевірка та завантаження файлів моделей..."):
    # Цей виклик кешується Streamlit, тому виконується лише один раз
    download_statuses = download_all_files(MODEL_IDS, PATHS)

# Виводимо статус, щоб користувач бачив, чи все завантажилося
with st.expander("Статус завантаження моделей"):
    for status in download_statuses.values():
        st.caption(status)


# ВСТАНОВЛЕННЯ ШЛЯХІВ У СЕСІЮ ДЛЯ UTILS
st.session_state['SVM_MODEL_PATH'] = PATHS["svm_model"]
st.session_state['SVM_VECTORIZER_PATH'] = PATHS["tfidf_vectorizer"]
st.session_state['BERT_MODEL_PATH'] = PATHS["bert_binary"]
st.session_state['BERT_MULTICLASS_PATH'] = PATHS["bert_multiclass"]
# Передаємо шлях до батьківської директорії SavedModel
st.session_state['CNN_MODEL_PATH'] = PATHS["cnn_model_dir"] 
# Примітка: Якщо хоча б один критичний файл CNN не завантажився, Keras викличе помилку
# при спробі завантажити модель у utils.initialize_models().


# --- UI: Шаблон Student hara-kiri ---

# Файл constants.py потрібен для роботи, створюємо його з mock-даними
try:
    import constants
except ImportError:
    st.error("Missing `constants.py` file. Creating a placeholder.")
    class Constants:
        introduction_text = "Ласкаво просимо до Student hara-kiri - багатомодельної системи виявлення тексту, згенерованого ШІ."
        version_info = "Оберіть, чи ви аналізуєте довге есе, чи короткий текст."
        threshold_info = "Поріг ймовірності, вище якого параграф вважається AI-генерованим."
        paraphrase_checker_info = "Увімкніть, щоб перевірити, чи був AI-генерований текст парафразований."
        paraphrase_threshold_info = "Поріг для визначення, чи є текст парафразованим."
        explanation_info = "Генерувати пояснення, які слова найбільше вплинули на класифікацію (потрібна модель SVM)."
    constants = Constants()

st.title("👨‍🎓 Student hara-kiri")

versions = ["Essay", "Short-Text"]

# layout
intro = st.container()
select_box_col, space = st.columns(2)
version = select_box_col.selectbox("Variation", versions, help=constants.version_info)
st.write("---")
predictor = st.container()
text_annotation = st.container()
chart = st.expander("Probability breakdown by each model")
explanability = st.expander("Explanability")

# Add introduction
with intro:
    st.markdown(constants.introduction_text)

# Set up the predictor layout
threshold_col, space, generate_explanation_col = predictor.columns(3)
text_to_check = predictor.text_area("Text to analyze", height=300)
check_col, reset_col = predictor.columns(2)
ai_score, paraphrased_score = predictor.columns(2)

threshold = threshold_col.slider(
    "Threshold", 0.0, 1.0, 0.5, 0.05, help=constants.threshold_info
)

check_paraphrase = generate_explanation_col.checkbox(
    "Check if text is paraphrased", help=constants.paraphrase_checker_info
)
if check_paraphrase:
    paraphrase_threshold = generate_explanation_col.slider(
        "Paraphase threshold",
        0.0,
        1.0,
        0.6,
        0.05,
        help=constants.paraphrase_threshold_info,
        key="paraphrase",
    )

generate_explanation = generate_explanation_col.checkbox(
    "Generate explanation", help=constants.explanation_info
)
if generate_explanation:
    model_selection = "SVM"
    number_of_features = generate_explanation_col.slider(
        "Number of features",
        10,
        100,
        20,
        1,
        key="explanation",
        help="Number of features to show in the explanation, the more features, the longer it takes to generate the explanation",
    )

# variable to check if the text is written by AI
written_by_ai = False
no_cyrillic = False
button_pressed = check_col.button(
    "Check if written by AI", disabled=len(text_to_check) == 0, type="primary"
)

# --- ЛІНИВЕ ЗАВАНТАЖЕННЯ ---
if button_pressed:
    with st.spinner("Initializing models and libraries... (First run may take a while)"):
        # Імпорт utils повинен бути тут, щоб дозволити завантаженню виконатися перед ініціалізацією моделей
        import utils 
    
    # Тепер utils доступний, і ми можемо його використовувати
    no_cyrillic = not utils.has_cyrillic(text_to_check)


if button_pressed and no_cyrillic and version == versions[0]:
    # utils вже імпортовано вище
    with st.spinner("Predicting..."):
        # check if the text is written by AI
        written_by_ai, scores, splitted_text = utils.check_if_ai(
            text_to_check, threshold
        )

        if check_paraphrase:
            is_paraphrased = utils.check_if_paraphrased(text_to_check)

        text_annotation.header("Text analysis")
        # generate the annotated text
        if not check_paraphrase:
            annotated_text_data = utils.generate_annotated_text(
                splitted_text, written_by_ai
            )
            with text_annotation:
                annotated_text(*annotated_text_data)

        # generate the chart
        df = pd.DataFrame.from_dict(scores["Average"], orient="index")
        fig = px.bar(
            df,
            orientation="h",
            labels={"index": "Model", "value": "Probability"},
            pattern_shape=df.index,
            color=df.index,
        )
        chart.plotly_chart(fig, use_container_width=True)

        # calculate the ai percentage
        ai_percentage = utils.check_ai_percentage(written_by_ai)
        ai_score.metric(
            label="AI",
            value=str(ai_percentage * 100)[:4] + "%",
            help="The percentage of the entire text that is written by AI",
        )

    # classify the text based on how many paragraphs are written by AI
    if ai_percentage > 0.8:
        ai_score.warning("The text is highly likely written by AI")
    elif ai_percentage > 0.6:
        ai_score.warning("The text is likely written by AI")
    elif ai_percentage > 0.4:
        ai_score.info("The text is may be written by AI")
    elif ai_percentage > 0.2:
        ai_score.success("The text is likely written by a human")
    else:
        ai_score.success("The text is most likely written by a human")

    if check_paraphrase:
        # generate the paraphrasing score
        paraphrasing_score = utils.check_if_paraphrased_percentage(
            is_paraphrased, paraphrase_threshold
        )

        annotated_text_data = utils.generate_annotated_text(
            splitted_text, written_by_ai, is_paraphrased, paraphrase_threshold
        )
        with text_annotation:
            annotated_text(*annotated_text_data)
        text_annotation.caption(
            "Legend: A red dotted border indicates that the text is paraphrased"
        )
        paraphrased_score.metric(
            label="Paraphrased",
            value=f"{paraphrasing_score*100}"[:4] + "%"
            if ai_percentage > 0.4
            else "N/A",
            help="The percentage of the entire text that is written by AI and paraphrased",
        )
        if ai_percentage > 0.4:
            # classify the text based on how many paragraphs are AI paraphrased
            if paraphrasing_score > 0.8:
                paraphrased_score.warning(
                    "The text is highly likely written by AI and paraphrased"
                )
            elif paraphrasing_score > 0.6:
                paraphrased_score.warning(
                    "The text is likely written by AI and paraphrased"
                )
            elif paraphrasing_score > 0.4:
                paraphrased_score.info(
                    "The text is may be written by AI and paraphrased"
                )
            elif paraphrasing_score > 0.2:
                paraphrased_score.success(
                    "The text is unlikely written by AI and paraphrased"
                )
        else:
            paraphrased_score.success(
                "The text is highly unlikely written by AI and paraphrased"
            )

    if generate_explanation:
        with st.spinner("Generating explanations..."):
            html = utils.get_explaination(
                text_to_check, number_of_features, model_selection
            )
            explanability._html(html, height=number_of_features * 45)

elif button_pressed and no_cyrillic and version == versions[1]:
    # utils вже імпортовано вище
    with st.spinner("Predicting..."):
        # check if the text is written by AI
        written_by_ai, scores = utils.check_if_ai_short_text(text_to_check, threshold)

        if check_paraphrase:
            is_paraphrased = utils.check_if_paraphrased_short_text(text_to_check)

        # generate the chart
        df = pd.DataFrame.from_dict(scores, orient="index")
        fig = px.bar(
            df,
            orientation="h",
            labels={"index": "Model", "value": "Probability"},
            pattern_shape=df.index,
            color=df.index,
        )
        chart.plotly_chart(fig, use_container_width=True)

        # calculate the ai percentage
        ai_percentage = 1 if written_by_ai == "AI" else 0
        ai_score.metric(
            label="AI",
            value=str(ai_percentage * 100)[:4] + "%",
            help="The percentage of the entire text that is written by AI",
        )

    # classify the text based on how many paragraphs are written by AI
    if ai_percentage > 0.8:
        ai_score.warning("The text is highly likely written by AI")
    elif ai_percentage > 0.6:
        ai_score.warning("The text is likely written by AI")
    elif ai_percentage > 0.4:
        ai_score.info("The text is may be written by AI")
    elif ai_percentage > 0.2:
        ai_score.success("The text is likely written by a human")
    else:
        ai_score.success("The text is most likely written by a human")

    if check_paraphrase:
        # generate the paraphrasing score
        paraphrasing_score = is_paraphrased[2]
        paraphrased_score.metric(
            label="Paraphrased",
            value=f"{paraphrasing_score*100}"[:4] + "%"
            if ai_percentage >= 0.4
            else "N/A",
            help="The percentage of the entire text that is written by AI and paraphrased",
        )
        # only show the paraphrasing score if the text is written by AI
        if ai_percentage >= 0.4:
            # classify the text based on how many paragraphs are AI paraphrased
            if paraphrasing_score > 0.8:
                paraphrased_score.warning(
                    "The text is highly likely written by AI and paraphrased"
                )
            elif paraphrasing_score > 0.6:
                paraphrased_score.warning(
                    "The text is likely written by AI and paraphrased"
                )
            elif paraphrasing_score > 0.4:
                paraphrased_score.info(
                    "The text is may be written by AI and paraphrased"
                )
            elif paraphrasing_score > 0.2:
                paraphrased_score.success(
                    "The text is unlikely written by AI and paraphrased"
                )
        else:
            paraphrased_score.success(
                "The text is highly unlikely written by AI and paraphrased"
            )

    if generate_explanation:
        with st.spinner("Generating explanations..."):
            html = utils.get_explaination(
                text_to_check, number_of_features, model_selection
            )
            explanability._html(html, height=number_of_features * 45)

elif button_pressed and not no_cyrillic:
    st.error("The text contains cyrillic characters, which is not supported by Student hara-kiri")

if reset_col.button("Reset"):
    written_by_ai = False
    button_pressed = False
    no_cyrillic = False