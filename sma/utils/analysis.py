import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_data(file):
    return pd.read_excel(file)


def plot_wordcloud(text, title, color='salmon'):
    stop_words = set(["https", "co", "RT"] + list(STOPWORDS))
    wordcloud = WordCloud(width=3000,
                          height=2000,
                          random_state=1,
                          background_color=color,
                          colormap='Pastel1',
                          collocations=False,
                          stopwords=stop_words).generate(text)
    # Plot
    plt.figure(figsize=(40, 30))
    # Display image
    plt.title(title)
    plt.imshow(wordcloud)
    plt.show()


def get_wordcloud(data):
    hate_cond = data['IsHate'].astype('Int64') == 1

    hate_text = ' '.join(data.loc[hate_cond, 'Text'].tolist())
    plot_wordcloud(hate_text, title='Hate Wordcloud', color='purple')

    non_hate_text = ' '.join(data.loc[~hate_cond, 'Text'].tolist())
    plot_wordcloud(non_hate_text, title='Non Hate Wordcloud')


if __name__ == '__main__':
    file = 'Combined_all_datasets_cleaned2.xlsx'

    data = load_data(file).dropna()
    get_wordcloud(data)

    print(data['IsHate'].value_counts())