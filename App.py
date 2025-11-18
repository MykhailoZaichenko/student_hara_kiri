import streamlit as st
import pandas as pd
import utils
import constants
import plotly.express as px
from annotated_text import annotated_text

st.set_page_config(page_title="Student hara-kiri", page_icon="👨‍🎓")

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

if button_pressed:
    no_cyrillic = not utils.has_cyrillic(text_to_check)


if button_pressed and no_cyrillic and version == versions[0]:
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
