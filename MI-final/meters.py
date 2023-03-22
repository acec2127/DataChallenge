from sklearn.metrics import adjusted_rand_score

class RandScore():

    def reset_train(self):
        self.sum_train = 0
        self.sum_squared_train = 0
        self.mean_train = 0
        self.var_train = 0
        self.n_train=0

    def reset_val(self):
        self.sum_val = 0
        self.sum_squared_val = 0
        self.mean_val = 0
        self.var_val = 0
        self.n_val = 0

    def add_train(self, target, pred):

        assert pred.shape == target.shape, (
            f"incompatible shape of `pred` and `target`, given "
            f"{pred.shape} and {target.shape}."
        )

        for i in range(target.shape[0]) :
            rand_score = adjusted_rand_score(target[i], pred[i])
            self.sum_train += rand_score
            self.sum_squared_train += rand_score ** 2
        self.n_train += target.shape[0]
        self.mean_train = self.sum_train /  self.n_train
        self.var_train = self.sum_squared_train /  self.n_train  - self.mean_train ** 2

    def add_val(self, target, pred):

        assert pred.shape == target.shape, (
            f"incompatible shape of `pred` and `target`, given "
            f"{pred.shape} and {target.shape}."
        )

        for i in range(target.shape[0]) :
            rand_score = adjusted_rand_score(target[i], pred[i])
            self.sum_val += rand_score
            self.sum_squared_val += rand_score ** 2
        self.n_val += target.shape[0]
        self.mean_val = self.sum_val /  self.n_val
        self.var_val = self.sum_squared_val /  self.n_val  - self.mean_val ** 2

    def summary(self) -> dict:
        # this function returns a dict and tends to aggregate the historical results.
        return {'Train' : {
            "Mean": self.mean_train, 
            "Var": self.var_train,
            },
            'Evaluation' : {
            "Mean": self.mean_val, 
            "Var": self.var_val,
            }
        }

    def __repr__(self):
        return f"Rand Score: {self.summary()}"