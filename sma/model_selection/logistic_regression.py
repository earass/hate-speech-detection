from sklearn.linear_model import LogisticRegression
from sma.model_selection import BaseModel


class LR(BaseModel):
    model_name = 'LogisticRegression'
    params_dict = dict(
        solver=['newton-cg', 'lbfgs', 'liblinear'],
        penalty=['l2'],
        C=[100, 10, 1.0, 0.1, 0.01]
    )

    def __init__(self, tune_params=False):
        super().__init__(model_name=LR.model_name)
        self.vectorize()
        self.model = LogisticRegression()
        if tune_params:
            self.tune_hyperparameters(param_grid=LR.params_dict)
        else:
            self.model = LogisticRegression(solver='liblinear',
                                            penalty='l2', C=1.0).fit(X=self.features_train, y=self.y_train)


if __name__ == '__main__':
    lr = LR(tune_params=False)
    lr.evaluate()
    lr.save_model()

