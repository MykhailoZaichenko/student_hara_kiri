import numpy as np
import pandas as pd

from classifiers.BertBaseClassifier import BertBaseClassifier
from classifiers.SvmTfidfBaseClassifier import SvmTfidfBaseClassifier
from classifiers.KerasModelClassifier import CNNGLTRClassifier
from models.ensemble.Ensemble import Ensemble # Виправлено імпорт на models.ensemble.Ensemble

# Константи шляхів, які використовувалися в utils (для консистентності)
SVM_MODEL_PATH = "models/svm_linear_model_90000_features_probability.pkl"
SVM_VECTORIZER_PATH = "models/tfidf_vectorizer_90000_features.pkl"
BERT_MODEL_PATH = "models/model_bertbase_updated.pt"
CNN_MODEL_PATH = "models/model_autokeras_gltr" # Припускаємо, що це папка

def predict(text):
    text_lst = [text]
    sample_df = pd.DataFrame(text_lst, columns=['response'])

    # Initialise models -- Adjust path accordingly
    bert = BertBaseClassifier(BERT_MODEL_PATH)
    # Змінено: Використовуємо SvmTfidfBaseClassifierV2, якщо це потрібно, або використовуємо 
    # SvmTfidfBaseClassifier з коректними шляхами для цієї версії.
    # Оскільки у вас був старий SvmTfidfBaseClassifier, я залишаю його, але використовую нові шляхи.
    svm_tfidf_classifier = SvmTfidfBaseClassifier(SVM_MODEL_PATH, SVM_VECTORIZER_PATH) 
    
    # Make sure you have the automodel file loaded locally as well
    cnn = CNNGLTRClassifier(CNN_MODEL_PATH)
    
    # Initialise ensemble
    models = [bert, svm_tfidf_classifier]
    ensemble = Ensemble(models, ["BERT","CNN","SVM"])

    # pred is an output "AI" or "Human" and output_dict shows what prediction each model made
    threshold = 0.6
    weights = np.array([0.25, 0.25,0.5])
    
    # Process input before prediction
    # Примітка: тут може знадобитися GLTRTransformer, який ви використовуєте в Utils.py
    # Якщо ви використовуєте main.py незалежно, то тут потрібен pipeline.
    # Але для прикладу залишимо як є, якщо логіка обробки даних була в utils.
    
    pred, output_dict = ensemble.predict(sample_df, weights, threshold)

    print(pred)
    print(output_dict)

ai_text = "Block modeling is like building a big puzzle! Imagine you have lots of puzzle pieces, but they are all different shapes and colors. Block modeling helps you group together pieces that are similar, so you can see how they fit together.In block modeling, you have a big picture with lots of dots and lines. Each dot is a person, and each line shows if two people are friends. But some people might have more friends than others, or might be friends with different kinds of people. So, block modeling helps you group together people who are similar, based on things like their age, gender, or hobbies.Once you group people together, you can see how the groups are connected to each other. It's like putting together puzzle pieces that are the same color, so you can see how they fit into the bigger picture. By using block modeling, you can understand how people are connected in the big picture of the social network, and how the groups of people are related to each other."
predict(ai_text)