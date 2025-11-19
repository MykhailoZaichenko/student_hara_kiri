import streamlit as st
# import utils  <-- ПРИБРАЛИ ЗВІДСИ
import constants
from annotated_text import annotated_text


st.set_page_config(page_title='Student hara-kiri', page_icon='🚀')

st.title('👨‍🎓 Student hara-kiri')

versions = ["Essay", "Short-Text"]

intro = st.container()
with intro:
    st.markdown(
        constants.model_2_introduction_text)
select_box_col, space = st.columns(2)
version = select_box_col.selectbox(
    "Variation", versions, help=constants.version_info)
st.write("---")
predictor = st.container()
text_annotation = st.container()
ai_score = st.container()

# enter the text to be cheked
text_to_check = predictor.text_area(
    'Text to analyze', height=300)

# variable to check if the text is written by AI
written_by_ai = False
no_cyrillic = False
check_col, reset_col = predictor.columns(2)

# Кнопка сама по собі не потребує спіннера, він потрібен при виконанні дії
button_pressed = check_col.button(
    'Check if written by AI', disabled=len(text_to_check) == 0, type='primary')

# --- ЛІНИВЕ ЗАВАНТАЖЕННЯ ---
if button_pressed:
    with st.spinner("Initializing models... (First run may take a while)"):
        import utils
    
    # Тепер utils доступний
    no_cyrillic = not utils.has_cyrillic(text_to_check)

if button_pressed and no_cyrillic and version == versions[0]:
    with st.spinner('Predicting...'):
        # check if the text is written by AI
        written_by_ai, splitted_text = utils.check_if_ai_speed(
            text_to_check)

        text_annotation.header("Text analysis")
        # generate the annotated text
        annotated_text_data = utils.generate_annotated_text_speed(
            splitted_text, written_by_ai)
        with text_annotation:
            annotated_text(
                *annotated_text_data
            )

        # calculate the ai percentage
        ai_percentage = utils.check_ai_percentage_speed(written_by_ai)
        ai_score.metric(label='AI', value=str(ai_percentage*100)[:4]+"%")

    # classify the text based on how many paragraphs are written by AI
    if ai_percentage > 0.8:
        ai_score.warning('The text is highly likely written by AI')
    elif ai_percentage > 0.6:
        ai_score.warning('The text is likely written by AI')
    elif ai_percentage > 0.4:
        ai_score.info('The text may be written by AI')
    elif ai_percentage > 0.2:
        ai_score.success('The text is likely written by a human')
    else:
        ai_score.success('The text is most likely written by a human')

if button_pressed and no_cyrillic and version == versions[1]:
    with st.spinner('Predicting...'):
        # check if the text is written by AI
        written_by_ai = utils.check_if_ai_speed_short_text(
            text_to_check)

        # calculate the ai percentage
        ai_percentage = written_by_ai
        ai_score.metric(label='AI', value=str(
            ai_percentage*100)[:4]+"%", help="The percentage of the entire text that is written by AI")

    # classify the text based on how many paragraphs are written by AI
    if ai_percentage > 0.8:
        ai_score.warning('The text is highly likely written by AI')
    elif ai_percentage > 0.6:
        ai_score.warning('The text is likely written by AI')
    elif ai_percentage > 0.4:
        ai_score.info('The text may be written by AI')
    elif ai_percentage > 0.2:
        ai_score.success('The text is likely written by a human')
    else:
        ai_score.success('The text is most likely written by a human')

elif button_pressed and not no_cyrillic:
    st.error(
        "The text contains cyrillic characters, which is not supported by Student hara-kiri")

if reset_col.button('Reset'):
    written_by_ai = False
    button_pressed = False
    no_cyrillic = False