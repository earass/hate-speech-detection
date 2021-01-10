from sqlalchemy import create_engine
from polyglot.detect import Detector
from sma.utils.credentials import db_user, db_pwd, host_ip
import matplotlib.pyplot as plt
import os
import pandas as pd

db = 'Twitter'


def db_con():
    return create_engine(f"mysql://{db_user}:{db_pwd}@{host_ip}/{db}")


def insert_tweets(df):
    con = db_con()
    df.to_sql(name='Tweets', schema=db, con=con, if_exists='append', index=False)
    con.dispose()
    print("Tweets inserted")


def detect_language(text):
    try:
        return str(Detector(text, quiet=True).language).split(" ")[1]
    except Exception as e:
        return "Unknown"


def line_plot(df, title):
    df.plot()
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def read_dataset():
    path = os.path.abspath(os.path.join('..', 'datasets', 'Combined_all_datasets_cleaned2.xlsx'))
    return pd.read_excel(path)
