from sma.utils import read_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os
import pickle
from sklearn.model_selection import GridSearchCV


class BaseModel:
    def __init__(self, model_name):
        self.df = BaseModel.read_dataset()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df['Text'], self.df['IsHate'],
                                                                                test_size=0.2, random_state=10)
        self.features_train = None
        self.features_test = None
        self.model = None
        self.model_name = model_name

    @classmethod
    def read_dataset(self):
        path = os.path.abspath(os.path.join('..', 'datasets', 'Combined_all_datasets_cleaned2.xlsx'))
        return pd.read_excel(path).dropna()

    def vectorize(self):
        tfidf = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
        self.features_train = tfidf.fit_transform(self.x_train)
        self.features_test = tfidf.transform(self.x_test)

    def tune_hyperparameters(self, param_grid):
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, n_jobs=-1, cv=10, scoring='accuracy',
                                   error_score=0)
        grid_result = grid_search.fit(self.features_train, self.y_train)
        print("Best score: ", grid_result.best_score_)
        print("Best params: ", grid_result.best_params_)

    def evaluate(self):
        y_pred = self.model.predict(self.features_test)
        print("Accuracy: ", accuracy_score(self.y_test, y_pred) * 100)
        print(classification_report(self.y_test, y_pred))

    def save_model(self):
        if self.model:
            pickle.dump(self.model, open(f"{self.model_name}.pickle", 'wb'))

    def load_model(self):
        try:
            return pickle.load(open(f"{self.model_name}.pickle", 'rb'))
        except:
            raise Exception("Model not found")

    def predict(self, X):
        if not self.model:
            my_model = self.load_model()
            return my_model.predict(X)
        else:
            return self.model.predict(X)
