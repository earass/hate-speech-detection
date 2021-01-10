from sklearn.naive_bayes import MultinomialNB
from sma.model_selection import BaseModel


class NB(BaseModel):
    model_name = 'MultinomialNB'
    params_dict = {'alpha': [1, 10, 0.1]}

    def __init__(self, tune_params=False):
        super().__init__(model_name=NB.model_name)
        self.vectorize()
        self.model = MultinomialNB()
        if tune_params:
            self.tune_hyperparameters(param_grid=NB.params_dict)
        else:
            self.model = MultinomialNB(alpha=1).fit(X=self.features_train, y=self.y_train)


if __name__ == '__main__':
    nb = NB(tune_params=False)
    nb.evaluate()
    nb.save_model()

