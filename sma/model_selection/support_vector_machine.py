from sklearn.svm import SVC
from sma.model_selection import BaseModel


class SVClassifier(BaseModel):
    model_name = 'SVC'
    params_dict = dict(
        gamma=[1.0, 0.1, 0.01],
        C=[100, 10, 1.0, 0.1, 0.01]
    )

    def __init__(self, tune_params=False):
        super().__init__(model_name=SVClassifier.model_name)
        self.vectorize()
        self.model = SVC()
        if tune_params:
            self.tune_hyperparameters(param_grid=SVClassifier.params_dict)
        else:
            self.model = SVC(gamma=0.1, C=10).fit(X=self.features_train, y=self.y_train)


if __name__ == '__main__':
    svc = SVClassifier(tune_params=False)
    svc.evaluate()
    svc.save_model()

