from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

model = load_model(f'{dir_path}/cls_model.h5')
model._make_predict_function()

with open(f'{dir_path}/tokenizer.pickle', 'rb') as handle:
    tkzr = pickle.load(handle)

categories = ['Normal', 'Hateful']


def get_prediction(text):
    encoded = tkzr.texts_to_sequences(text)
    padded_docs = pad_sequences(encoded, maxlen=100, padding='post')
    pred_classes = model.predict_classes(padded_docs, verbose=1)
    predictions = [i[0] for i in pred_classes]
    return predictions
