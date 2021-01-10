import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_data(file):
    return pd.read_excel(file)


def remove_tags(text):
    new = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", text).split())
    return new


def remove_stopwords(text):
    stop_words = set(stopwords.words('english') + ['RT'])
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    sent = ' '.join(filtered_sentence)
    return sent


def cleaning(data):

    # removed tags
    data['Text'] = data['Text'].astype(str).apply(remove_tags)

    # remove stopwords
    data['Text'] = data['Text'].astype(str).apply(remove_stopwords)

    data = data.loc[data["Text"] != '']

    return data


if __name__ == '__main__':
    inp = 'Combined_all_datasets2.xlsx'
    out = 'Combined_all_datasets_cleaned2.xlsx'

    data = load_data(inp).dropna()
    data = cleaning(data)
    data.to_excel(out)

