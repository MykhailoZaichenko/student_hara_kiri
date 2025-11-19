import streamlit as st
import gdown
import os
import zipfile 
# Моделі тепер будуть завантажені у `utils` одразу при імпорті.

# --- Налаштування сторінки ---
st.set_page_config(
    page_title="AI Text Detector (BERT, CNN, SVM)",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Налаштування шляхів та ID ---
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

# !!! УВАГА: ЗАМІНІТЬ ЦЕЙ PLACEHOLDER НА РЕАЛЬНИЙ ID ВАШОГО ZIP-АРХІВУ CNN !!!
# Якщо ви не заміните цей ID, модель CNN не буде завантажена.
CNN_MODEL_ZIP_ID = '1lLGHDE0o_aJyUOVbJ37fspImQRKKTrjA'
CNN_MODEL_ZIP_FILENAME = "cnn_model.zip"
SAVED_MODEL_FILE = 'saved_model.pb' # Ключовий файл для Keras

# ID файлів моделей з Google Drive
MODEL_IDS = {
    "svm_model": "1_O8mtsgJipuCgqrW1yBoJBUUEYCiUXsx", # svm_linear_model_90000_features_probability.pkl
    "tfidf_vectorizer": "1HBpIvydoh6slZKwrX4o9hsinGPydwkIi", # tfidf_vectorizer_90000_features.pkl
    "bert_binary": "1D8wp3sOVV9Ri5BUG26IGVSZoSZlvjobD", # model_bertbase_updated.pt
    "bert_multiclass": "1GhTr-2ghquSTWdha96s7JJWegx2yoo2t", # model_multiclass.pt
    "cnn_zip": CNN_MODEL_ZIP_ID, # ID для ZIP-файлу моделі CNN
}

# Шляхи до файлів (використовуємо лише для завантаження, шлях до CNN буде знайдено динамічно)
PATHS = {
    "svm_model": os.path.join(MODELS_DIR, "svm_linear_model_90000_features_probability.pkl"),
    "tfidf_vectorizer": os.path.join(MODELS_DIR, "tfidf_vectorizer_90000_features.pkl"),
    "bert_binary": os.path.join(MODELS_DIR, "model_bertbase_updated.pt"),
    "bert_multiclass": os.path.join(MODELS_DIR, "model_multiclass.pt"),
    "cnn_zip": os.path.join(MODELS_DIR, CNN_MODEL_ZIP_FILENAME),
    "cnn_model_dir": None, # Цей шлях буде визначено динамічно після розпакування
}


# --- Функція для завантаження файлів моделей (виконується один раз при старті) ---
def download_models(model_paths):
    """Завантажує файли моделей з Google Drive, якщо вони відсутні."""
    
    # Спеціальна обробка для CNN SavedModel (ZIP)
    cnn_zip_path = model_paths["cnn_zip"]
    cnn_zip_id = MODEL_IDS["cnn_zip"]
    
    # Крок 1: Перевірка, чи не знайдена модель вже була раніше (для кешування)
    final_cnn_path = None
    for root, dirs, files in os.walk(MODELS_DIR):
        if SAVED_MODEL_FILE in files:
            final_cnn_path = root
            break
            
    if final_cnn_path:
        # Модель вже завантажена та знайдена, пропускаємо завантаження
        st.success(f"CNN/GLTR model found and ready at: {final_cnn_path}!")
        model_paths["cnn_model_dir"] = final_cnn_path
        
    elif cnn_zip_id != '1lLGHDE0o_aJyUOVbJ37fspImQRKKTrjA':
        # Якщо модель не знайдена і ID встановлено, спробуємо завантажити
        with st.spinner("Downloading and setting up CNN/GLTR model (SavedModel ZIP)..."):
            try:
                # 1. Завантаження ZIP
                # gdown.download повертає шлях, якщо успішно
                result_path = gdown.download(f'https://drive.google.com/uc?id={cnn_zip_id}', cnn_zip_path, quiet=False)
                
                if not result_path:
                    # Якщо gdown не повернув шлях, завантаження не відбулося
                    raise Exception("gdown failed to download the file. Check the Google Drive ID and file permissions (must be 'Anyone with the link').")
                    
                st.info("CNN ZIP downloaded successfully. Starting extraction...")
                
                # 2. Розпакування
                with zipfile.ZipFile(cnn_zip_path, 'r') as zip_ref:
                    # Розпаковуємо безпосередньо в каталог models.
                    zip_ref.extractall(MODELS_DIR) 
                
                # 3. Рекурсивний пошук коректного шляху для Keras (SavedModel)
                found_keras_path = None
                for root, dirs, files in os.walk(MODELS_DIR):
                    if SAVED_MODEL_FILE in files:
                        found_keras_path = root
                        break
                
                if found_keras_path:
                    # Присвоюємо знайдений шлях
                    model_paths["cnn_model_dir"] = found_keras_path
                    st.success(f"CNN/GLTR model extracted and ready at: {found_keras_path}!")
                else:
                    # Якщо файл SavedModel не знайдено, це серйозна помилка
                    raise FileNotFoundError(f"Cannot find '{SAVED_MODEL_FILE}' inside the extracted ZIP content in {MODELS_DIR}. Check ZIP file structure.")

                # 4. Видалення ZIP-файлу
                os.remove(cnn_zip_path)
                
            except Exception as e:
                # Виводимо детальну помилку, якщо щось пішло не так
                st.error(f"❌ FATAL ERROR: Помилка обробки CNN моделі. Перевірте ID та доступ до Google Drive. Деталі: {e}")
                st.session_state['cnn_error_setup'] = str(e)

    elif cnn_zip_id == '1lLGHDE0o_aJyUOVbJ37fspImQRKKTrjA':
        # Випадок, коли ID не замінено
        st.error("🚨 ВАЖЛИВО: ID моделі CNN не встановлено. Будь ласка, замініть '1lLGHDE0o_aJyUOVbJ37fspImQRKKTrjA' на реальний ID Google Drive в App.py. Модель CNN буде недоступна.")


    # Завантажуємо інші, поодинокі файли (як і раніше)
    for key in ["svm_model", "tfidf_vectorizer", "bert_binary", "bert_multiclass"]:
        path = model_paths[key]
        if key in MODEL_IDS and not os.path.exists(path):
            with st.spinner(f"Downloading {key}..."):
                try:
                    gdown.download(f'https://drive.google.com/uc?id={MODEL_IDS[key]}', path, quiet=False)
                    st.success(f"{key} downloaded successfully!")
                except Exception as e:
                    st.error(f"Error downloading {key}: {e}")
                    st.session_state[f'{key}_error_setup'] = str(e)


# --- ФАЗА 1: ЗАВАНТАЖЕННЯ ФАЙЛІВ ПРИ СТАРТІ ЗАСТОСУНКУ ---
download_models(PATHS)

# ВСТАНОВЛЕННЯ ШЛЯХІВ У СЕСІЮ ДЛЯ UTILS
st.session_state['SVM_MODEL_PATH'] = PATHS["svm_model"]
st.session_state['SVM_VECTORIZER_PATH'] = PATHS["tfidf_vectorizer"]
st.session_state['BERT_MODEL_PATH'] = PATHS["bert_binary"]
st.session_state['BERT_MULTICLASS_PATH'] = PATHS["bert_multiclass"]
# Передаємо динамічно знайдений шлях (або None, якщо завантаження не вдалося)
st.session_state['CNN_MODEL_PATH'] = PATHS["cnn_model_dir"] 

# --- ФАЗА 2: ІМПОРТ UTILS І ІНІЦІАЛІЗАЦІЯ МОДЕЛЕЙ ---
# Моделі ініціалізуються (і кешуються Streamlit) в utils.py одразу при імпорті.
with st.spinner("Ініціалізація моделей та бібліотек... (Виконується лише при першому запуску)"):
    import utils
    
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

# --- Основна логіка: Запуск аналізу ---

if button_pressed:
    if not text_to_check.strip():
        st.warning("Будь ласка, вставте текст для аналізу.")
    else:
        # Аналіз тепер буде миттєвим, оскільки моделі вже ініціалізовані
        with st.spinner("Виконання аналізу..."):
            try:
                
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