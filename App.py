import streamlit as st
import gdown
import os
# Необхідний імпорт для роботи з архівами
import zipfile 

# --- Налаштування сторінки ---
st.set_page_config(
    page_title="AI Text Detector (BERT, CNN, SVM)",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Налаштування шляхів та ID ---
# Шляхи до моделей (використовуємо один і той самий каталог)
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

# !!! УВАГА: ЗАМІНІТЬ ЦЕЙ PLACEHOLDER НА РЕАЛЬНИЙ ID ВАШОГО ZIP-АРХІВУ CNN !!!
CNN_MODEL_ZIP_ID = '1lLGHDE0o_aJyUOVbJ37fspImQRKKTrjA'
CNN_MODEL_ZIP_FILENAME = "cnn_model.zip"

# ID файлів моделей з Google Drive
MODEL_IDS = {
    "svm_model": "1_O8mtsgJipuCgqrW1yBoJBUUEYCiUXsx", # svm_linear_model_90000_features_probability.pkl
    "tfidf_vectorizer": "1HBpIvydoh6slZKwrX4o9hsinGPydwkIi", # tfidf_vectorizer_90000_features.pkl
    "bert_binary": "1D8wp3sOVV9Ri5BUG26IGVSZoSZlvjobD", # model_bertbase_updated.pt
    "bert_multiclass": "1GhTr-2ghquSTWdha96s7JJWegx2yoo2t", # model_multiclass.pt
    "cnn_zip": CNN_MODEL_ZIP_ID, # ID для ZIP-файлу моделі CNN
}

# Шляхи до файлів
PATHS = {
    "svm_model": os.path.join(MODELS_DIR, "svm_linear_model_90000_features_probability.pkl"),
    "tfidf_vectorizer": os.path.join(MODELS_DIR, "tfidf_vectorizer_90000_features.pkl"),
    "bert_binary": os.path.join(MODELS_DIR, "model_bertbase_updated.pt"),
    "bert_multiclass": os.path.join(MODELS_DIR, "model_multiclass.pt"),
    "cnn_zip": os.path.join(MODELS_DIR, CNN_MODEL_ZIP_FILENAME),
    "cnn_model_dir": os.path.join(MODELS_DIR, "model_autokeras_gltr_trials_8"), # Кінцевий шлях для CNN
}

# --- Функція для завантаження файлів моделей ---
def download_models(model_paths):
    """Завантажує файли моделей з Google Drive, якщо вони відсутні."""
    
    # Спочатку завантажуємо та обробляємо CNN SavedModel (ZIP)
    cnn_target_dir = PATHS["cnn_model_dir"]
    cnn_zip_path = PATHS["cnn_zip"]
    cnn_zip_id = MODEL_IDS["cnn_zip"]
    
    if not os.path.exists(cnn_target_dir) and cnn_zip_id != '1lLGHDE0o_aJyUOVbJ37fspImQRKKTrjA':
        with st.empty():
            st.info("Downloading CNN/GLTR model (SavedModel ZIP)...")
            try:
                # 1. Завантаження ZIP
                gdown.download(f'https://drive.google.com/uc?id={cnn_zip_id}', cnn_zip_path, quiet=False)
                st.success("CNN ZIP downloaded successfully. Starting extraction...")
                
                # 2. Розпакування
                with zipfile.ZipFile(cnn_zip_path, 'r') as zip_ref:
                    # Розпаковуємо безпосередньо в каталог models, 
                    # припускаючи, що cnn_model.zip містить папку model_autokeras_gltr_trials_8
                    zip_ref.extractall(MODELS_DIR) 
                
                # 3. Видалення ZIP-файлу
                os.remove(cnn_zip_path)
                st.success(f"CNN/GLTR model extracted and ready at {cnn_target_dir}!")
                
            except Exception as e:
                st.error(f"Error processing CNN model (ZIP/Extraction): {e}. Check if the ZIP file contains the directory 'model_autokeras_gltr_trials_8'.")
                st.session_state['cnn_error_setup'] = str(e)
                # Ми не зупиняємо, щоб можна було перевірити інші моделі
    elif cnn_zip_id == '1lLGHDE0o_aJyUOVbJ37fspImQRKKTrjA':
        st.warning("Будь ласка, оновіть App.py з реальним Google Drive ID для моделі CNN.")

    
    # Тепер завантажуємо інші, поодинокі файли
    for key in ["svm_model", "tfidf_vectorizer", "bert_binary", "bert_multiclass"]:
        path = model_paths[key]
        if key in MODEL_IDS and not os.path.exists(path):
            with st.empty():
                try:
                    st.info(f"Downloading {key}...")
                    gdown.download(f'https://drive.google.com/uc?id={MODEL_IDS[key]}', path, quiet=False)
                    st.success(f"{key} downloaded successfully!")
                except Exception as e:
                    st.error(f"Error downloading {key}: {e}")
                    st.session_state[f'{key}_error_setup'] = str(e)

# --- UI ---
st.title("🔎 Мультимодельний Детектор Тексту, Згенерованого Штучним Інтелектом (AI)")

text_to_check = st.text_area(
    "Вставте текст для аналізу:",
    height=300,
    value="Це просто звичайний тестовий текст, написаний людиною, щоб перевірити, як працює ваша система виявлення ШІ. Чи зможе вона мене розпізнати?",
    key="text_input"
)

st.caption("Система перевіряє текст на оригінальність за допомогою чотирьох різних моделей: SVM/TF-IDF, бінарний BERT, багатокласовий BERT та CNN/GLTR.")

# Кнопка для запуску перевірки
button_pressed = st.button("Перевірити на AI", type="primary")

# --- Основна логіка: Завантаження моделей та запуск аналізу ---

# Завантажуємо всі файли. Функція тепер обробляє ZIP для CNN.
download_models(PATHS)

if button_pressed:
    if not text_to_check.strip():
        st.warning("Будь ласка, вставте текст для аналізу.")
    else:
        # Імпорт utils відбувається тут, щоб ініціалізація моделей була лінивою
        with st.spinner("Ініціалізація моделей та бібліотек... (Перший запуск може зайняти до хвилини)"):
            try:
                # ВСТАНОВЛЕННЯ ШЛЯХІВ У СЕСІЮ ДЛЯ UTILS
                st.session_state['SVM_MODEL_PATH'] = PATHS["svm_model"]
                st.session_state['SVM_VECTORIZER_PATH'] = PATHS["tfidf_vectorizer"]
                st.session_state['BERT_MODEL_PATH'] = PATHS["bert_binary"]
                st.session_state['BERT_MULTICLASS_PATH'] = PATHS["bert_multiclass"]
                # Шлях до директорії SavedModel
                st.session_state['CNN_MODEL_PATH'] = PATHS["cnn_model_dir"] 
                
                import utils
                
                # Тепер utils доступний, і ми можемо його використовувати
                no_cyrillic = not utils.has_cyrillic(text_to_check)
                
                # Перевірка на наявність кирилиці
                if no_cyrillic:
                    st.error("⚠️ Увага: Введений текст не містить кирилиці. Система оптимізована для української та російської мов. Результати для тексту англійською або іншою мовою можуть бути неточними.")
                
                # Запуск аналізу
                results, multiclass_results = utils.run_analysis(text_to_check)
                
                # --- Виведення Результатів ---
                st.header("📊 Зведений Результат Аналізу")
                
                # Створення колонок для зведення
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Бінарна Класифікація (AI / Human)")
                    for model_name, (is_ai, prob) in results.items():
                        # Пропускаємо CNN, якщо вона не завантажилася
                        if prob is None:
                            continue
                        
                        st.metric(
                            label=model_name,
                            value="AI 🤖" if is_ai else "Людина ✍️",
                            delta=f"Впевненість: {prob:.2f}%",
                            # Використовуємо інверсний колір для AI
                            delta_color="inverse" if is_ai else "normal"
                        )

                with col2:
                    st.subheader("Багатокласова Класифікація (Multiclass BERT)")
                    if multiclass_results:
                        for label, prob in multiclass_results.items():
                            st.metric(
                                label=label,
                                value=f"{prob:.2f}%",
                                delta_color="off"
                            )
                    else:
                        st.info("Результати багатокласової класифікації недоступні.")


                st.header("🔬 Детальний Аналіз")

                with st.expander("Пояснення моделей"):
                    st.markdown("""
                    **SVM/TF-IDF:** Класична модель машинного навчання, яка базується на частоті слів та їхніх комбінацій. Швидка, але менш точна.  
                    **BERT (Бінарний):** Модель глибокого навчання, налаштована для визначення, чи є текст AI-генерованим (ChatGPT/GPT-3).  
                    **BERT (Багатокласовий):** Намагається визначити конкретного генератора (ChatGPT, GPT-3, T5, LLaMA).  
                    **CNN/GLTR:** Навчалася на ознаках, пов'язаних з тим, як ШІ обирає наступне слово (GLTR). 
                    """)
                    
                # Додамо елемент, щоб показати, що CNN не завантажилась
                if 'cnn_error' in st.session_state:
                    st.warning(f"Помилка при ініціалізації CNN/GLTR: {st.session_state['cnn_error']}")


            except Exception as e:
                st.error(f"❌ Сталася помилка під час аналізу. Спробуйте ще раз або перевірте введений текст.")
                st.exception(e)