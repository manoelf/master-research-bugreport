
class TainingModels:

    def __init__(self):
        pass


    def train_model(self, model, x_train, y_train):
        model = model.fit(x_train, y_train)
        return model


    def predict_model(self, model, x_test):
        pred = model.predict(x_test)
        return pred
