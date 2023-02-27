import pandas as pd


class pandas_encoder:
    def __init__(self):
        self.data = None
        self.__decoder = {}

    
    def fit(self, x: [pd.Series, pd.DataFrame]):
        if type(x) in [pd.Series, pd.DataFrame]:
            self.data = x
        else:
            raise Exception(f"Unsupported type {type(x)}!")


    def __series_encode(self, x):
        v = list(dict.fromkeys(x.to_list()))
        en = {}
        for i in range(len(v)):
            en[i] = v[i]
        # den = {k: v for k, v in en.items()}
        return [list(en.values()).index(j) for j in x.to_list()], en


    def encode(self):
        if type(self.data) == pd.Series:
            processed = self.__series_encode(self.x)
            self.__decoder = processed[1]
            return processed[0]
        elif type(self.data) == pd.DataFrame:
            td = self.data.copy()
            for k in self.data.columns.to_list():
                processed = self.__series_encode(td[k])
                self.__decoder[k] = processed[1]
                td[k] = processed[0]
            return td
        else:
            raise Exception(f"Cannot encode the data with type {type(self.data)}!")

    
    def decode(self, x):
        if type(x) == pd.Series:
            return [self.__decoder[k] for k in x.to_list()]
        elif type(x) == pd.DataFrame:
            td = x.copy()
            for k in x.columns.to_list():
                try:
                    td[k] = [self.__decoder[k][i] for i in x[k].to_list()]
                except:
                    td[k] = x[k].to_list()
            return td
        else:
            raise Exception(f"Cannot encode the data with type {type(x)}!")