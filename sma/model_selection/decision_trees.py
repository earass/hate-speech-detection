from sklearn.tree import DecisionTreeClassifier
from sma.model_selection import BaseModel


class DTC(BaseModel):
    model_name = 'DecisionTreeClassifier'
    params_dict = dict(
        criterion=['gini', 'entropy'],
        min_samples_split=range(1, 10),
        max_depth=range(1, 10))

    def __init__(self, tune_params=False):
        super().__init__(model_name=DTC.model_name)
        self.vectorize()
        self.model = DecisionTreeClassifier()
        if tune_params:
            self.tune_hyperparameters(param_grid=DTC.params_dict)
        else:
            self.model = DecisionTreeClassifier(criterion='gini',
                                                min_samples_split=5,
                                                max_depth=9).fit(X=self.features_train, y=self.y_train)


if __name__ == '__main__':
    dtc = DTC(tune_params=False)
    dtc.evaluate()
    dtc.save_model()

