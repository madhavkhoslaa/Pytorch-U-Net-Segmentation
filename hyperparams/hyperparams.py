class hyperparameters:
    def __init__(self, **kwargs):
        self.hyperparameters= dict()
        for key, value in kwargs.items():
            self.hyperparameters.update({key: value})

