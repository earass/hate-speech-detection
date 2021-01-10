import tweepy
from tweepy import OAuthHandler
from sma.utils.credentials import twitter_consumer_key, twitter_consumer_secret_key, twitter_access_token, \
    twitter_access_token_secret
import pandas as pd
from sma.utils.clean import cleaning
from sma.utils.analysis import get_wordcloud
from sma.classifier.predict import get_prediction
from sma.utils import insert_tweets, detect_language, line_plot


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

auth = OAuthHandler(twitter_consumer_key, twitter_consumer_secret_key)
auth.set_access_token(twitter_access_token, twitter_access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def fetch_tweets(screen_name):
    pages = []

    user_id = None
    for page in tweepy.Cursor(api.user_timeline, user_id=user_id, screen_name=screen_name,
                              count=200, tweet_mode='extended',
                              include_rts=True).pages(16):
        pages.extend([i._json for i in page])

    df = pd.DataFrame(pages)
    df['UserName'] = screen_name
    return df


def clean_data(df):
    df['Text'] = df['full_text']
    df = cleaning(df)
    df['CreatedAt'] = pd.to_datetime(df['created_at'])
    df['Year'] = df['CreatedAt'].dt.year
    df['Month'] = df['CreatedAt'].dt.month_name()
    df['MonthYear'] = df['Year'].astype(str) + " " + df['Month']
    df['Language'] = df['Text'].apply(detect_language)
    return df


def assign_labels(df):
    lang_cond = df['Language'] == 'English'
    df.loc[lang_cond, 'IsHate'] = get_prediction(df.loc[lang_cond, 'Text'].to_list())
    return df


def get_monthly_trend(df):
    trend = df.groupby(['MonthYear', 'IsHate']).agg(
            count=('id', 'count')
        ).sort_values('count', ascending=False).reset_index().pivot(index='MonthYear', columns='IsHate', values='count')
    line_plot(trend, title='Volume by Month')


def execute(screen_name, insert=True):
    df = fetch_tweets(screen_name)
    print(df.head())
    df = clean_data(df)
    df = assign_labels(df)
    # get_wordcloud(df)
    get_monthly_trend(df)
    df = df[['id', 'UserName', 'Text', 'IsHate', 'user', 'CreatedAt', 'Year', 'Month', 'Language']]
    if insert:
        insert_tweets(df)
