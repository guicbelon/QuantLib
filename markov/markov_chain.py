from quantitiative_finances import Database
import pandas as pd
import numpy as np
from datetime import date, timedelta
from bisect import bisect

class MarkovChain:
    def __init__(
        self,
        ticker: str,
        close_date: str,
        period: int = 20,
        num_of_scales: int = 3,
        filter: float = 0.97,
        open_date: str = None,
        close_date_of_test: str = None
    ) -> None:
        self.ticker = ticker
        self.period = period
        self.num_of_scales = num_of_scales
        self.filter = filter
        self.open_date = open_date
        self.close_date = close_date
        self.close_date_of_test = close_date_of_test
        self.df_complete = Database().get_info(
            ticker,
            open_date=self.open_date,
            close_date=self.close_date_of_test)
        self.df = self.df_complete.loc[self.open_date:self.close_date]
        self.df_test = self.df_complete

    def create_configuration(self, df, num_of_scales=3, period=20):
        values = (df[df.columns[-1]])
        rol_mean = values.rolling(period).mean()
        rol_dev_pad = values.rolling(period).std()
        ratio = ((values-rol_mean)/rol_dev_pad).dropna()
        sorted_ratio = ratio.sort_values()
        min_value, max_value = sorted_ratio[0], sorted_ratio[-1]
        down_spacement = np.linspace(min_value, 0, num_of_scales+1)
        growth_spacement = np.linspace(0, max_value, num_of_scales+1)
        intervals = np.concatenate((down_spacement[:-1], growth_spacement))
        down, growth = [], []
        for scale in range(1, num_of_scales+1):
            down.append('d'+str(num_of_scales-scale+1))
            growth.append('g'+str(scale))
        states = down+growth
        configuration = []
        for value in ratio:
            index = bisect(intervals, value)
            if index > len(states):
                index -= 1
            configuration.append(states[index-1])
        return configuration

    def create_transition_matrix(self, configuration):
        df = pd.DataFrame(configuration)
        df['shift'] = df[0].shift(-1)
        df['count'] = 1
        trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
        trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values
        return trans_mat

    def get_states_to_trade(self):
        configuration = self.create_configuration(
            self.df, self.num_of_scales, self.period)
        matrix = self.create_transition_matrix(configuration)
        down = []
        grown = []
        for index in range(1, self.num_of_scales+1):
            down.append('d'+str(index))
            grown.append('g'+str(index))
        possible_states = down+grown
        trades = []
        for row_index in range(len(matrix)):
            row = matrix[row_index]
            down_part = sum(row[:self.num_of_scales])
            grow_part = sum(row[self.num_of_scales:])
            if down_part >= self.filter or grow_part >= self.filter:
                trades.append(possible_states[row_index])
        return set(trades)

    def get_current_state(self):
        date_to_predict = self.close_date_of_test
        df_with_date_to_predict = Database().get_info(
            tickers=self.ticker,
            open_date=date_to_predict-timedelta(self.period+100),
            close_date=date_to_predict)
        config = self.create_configuration(
            df=df_with_date_to_predict, period=self.period, num_of_scales=self.num_of_scales)
        print(config)
        return config[-1]
