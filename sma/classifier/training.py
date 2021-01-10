import pandas as pd
from numpy import asarray, zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Embedding, Dropout, Dense
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sma.utils import read_dataset


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('ROC_curve')
    plt.show()


def train_model():
    # importing the dataset as pandas dataframe
    df = read_dataset()

    df.dropna(inplace=True)
    print(df.shape)

    texts = df['Text']
    labels = df['IsHate']

    # tokenizing
    tkzr = Tokenizer()
    tkzr.fit_on_texts(texts)
    words_dict = tkzr.word_index
    vocab_size = len(words_dict) + 1  # number of unique words

    # saving vocabulary
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tkzr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # integer encoding the texts, assigning integers to each text in texts
    encoded_texts = tkzr.texts_to_sequences(texts)

    # padding each sequence to the same length
    max_len = 100
    padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding='post')

    # splitting dataset into train set (0.8) and test set (0.2)
    X_train, X_test, y_train, y_test = train_test_split(padded_texts, labels, random_state=0, test_size=0.2)

    # loading the pre-trained Glove embeddings
    embeddings_index = dict()
    f = open('glove.twitter.27B/glove.twitter.27B.100d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        if words_dict.get(word):  # filtering it only for the unique words in our training data
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()

    # creating a weight matrix for words in training texts
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in words_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # defining model
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fiting the model
    model.fit(X_train, y_train, epochs=1, verbose=1)

    # evaluating the model on test set
    accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(accuracy[1] * 100)

    pred = model.predict_classes(X_test)
    print(pred)
    print(classification_report(y_true=y_test, y_pred=pred))

    # saving the model
    model.save('cls_model.h5')
    print('model saved')

    # keep probabilities for the positive outcome only
    probs = model.predict_proba(X_test)

    auc = roc_auc_score(y_test, probs)
    print(f'AUC: {auc}')

    # calculate scores
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plot_roc_curve(fpr, tpr)


if __name__ == '__main__':
    train_model()
