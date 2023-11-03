import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from datetime import timedelta, date
from ..constants import *
from ..data_management import *
from .model_database import ModelDatabase


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MLPModel():
    def __init__(self,
                 open_date: str = None,
                 close_date: str = None,
                 train_df: pd.DataFrame = None,
                 test_df: pd.DataFrame = None) -> None:
        if close_date is None:
            close_date = pd.to_datetime(date.today())
        if open_date is None:
            open_date = close_date - timedelta(days=5*365)
        self.open_date = open_date
        self.close_date = close_date
        self.database_model = ModelDatabase(
            open_date=self.open_date,
            close_date=self.close_date,
            remove_outliers_returns=True)
        self.json_file_name = 'files/df_data.json'
        self.scaler = StandardScaler()
        self.model = None
        self.ticker = None
        self.predicted_df = None
        self.train_df = train_df
        self.test_df = test_df

    def set_ticker(self, ticker: str):
        self.ticker = ticker
        self.database_model.set_ticker(ticker)
        self.predicted_df = None
        self.train_df = None
        self.test_df = None

    def mlp(self):
        if self.model is not None:
            return self.model
        mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                                max_iter=300, activation='relu',
                                solver='adam')
        self.model = mlp_clf
        return self.model

    def train(self, use_equal_distributed: bool = True):
        if self.train_df is None:
            self.train_df = self.database_model.create_training_database()
        if use_equal_distributed:
            self.train_df = self.database_model.create_equal_distributed_database(
                self.train_df, column_name='config')
        y_data = self.train_df[['config']]
        x_data = self.train_df.drop(columns=['config'])
        x_data = self.scaler.fit_transform(x_data)
        model = self.mlp()
        model.fit(x_data, y_data)
        dump(model, f'files/models/{self.ticker}.joblib')

    def test(self):
        if self.predicted_df is not None:
            return self.predicted_df
        if self.test_df is None:
            self.test_df = self.database_model.create_test_database()
        self.test_df = self.database_model.create_test_database()
        self.correct_test_df = self.test_df[['config']]
        x_data = self.test_df.drop(columns=['config'])
        x_data = self.scaler.fit_transform(x_data)
        model = load(f'files/models/{self.ticker}.joblib')
        y_predicted = model.predict(x_data)
        df_predict = pd.DataFrame(y_predicted, columns=[
                                  'predict'], index=self.correct_test_df.index)
        self.predicted_df = df_predict
        return df_predict

    def get_current_state(self):
        predicted = self.test()
        return predicted.values[-1][0]

    def get_metrics(self):
        predicted_df = self.test()
        original_df = self.correct_test_df
        dates = original_df.index
        dates_predicted_g3 = dates[(np.where(predicted_df == 'g3')[0])]
        dates_predicted_g2 = dates[(np.where(predicted_df == 'g2')[0])]
        overall_accuracy = np.sum(
            original_df.values == predicted_df.values)/len(original_df)
        predicted_g3 = len(dates_predicted_g3)
        correctly_predicted_g3 = sum(
            original_df.loc[dates_predicted_g3]['config'].str.get(0) == 'g')
        if predicted_g3 == 0:
            g3_accuracy = 0
        else:
            g3_accuracy = correctly_predicted_g3/predicted_g3
        predicted_g2 = len(dates_predicted_g2)
        correctly_predicted_g2 = sum(
            original_df.loc[dates_predicted_g2]['config'].str.get(0) == 'g')
        if predicted_g2 == 0:
            g2_accuracy = 0
        else:
            g2_accuracy = correctly_predicted_g2/predicted_g2
        g3_g2_accuracy = (correctly_predicted_g3 +
                          correctly_predicted_g2)/max(predicted_g3+predicted_g2, 1)
        info = {
            'overall_accuracy': overall_accuracy,
            'g3_accuracy': g3_accuracy,
            'g2_accuracy': g2_accuracy,
            'g3_g2_accuracy': g3_g2_accuracy,
            'predicted_g3': predicted_g3,
            'correctly_predicted_g3': correctly_predicted_g3,
            'predicted_g2': predicted_g2,
            'correctly_predicted_g2': correctly_predicted_g2,
        }
        return info
