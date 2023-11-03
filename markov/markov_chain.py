from ..database import Database
from ..constants import *
from ..data_management import *
import pandas as pd
import numpy as np
from datetime import date, timedelta
from bisect import bisect


class MarkovChain:
    def __init__(
        self,
        ticker: str,
        close_date: str,
        num_of_scales: int = None,
        filter: float = 0.97,
        open_date: str = None,
        close_date_of_test: str = None,
        remove_outliers: bool = True,
        use_returns: bool = False,
        period: int = 3
    ) -> None:
        if use_returns:
            ticker = 'RET_'+ticker
        self.ticker = ticker
        if num_of_scales is None:
            num_of_scales = NUM_OF_SCALES
        self.num_of_scales = num_of_scales
        self.filter = filter
        self.use_returns = use_returns
        self.period = period
        self.open_date = open_date
        self.close_date = close_date
        self.close_date_of_test = close_date_of_test
        df_complete = Database().get_info(
            ticker,
            open_date=self.open_date,
            close_date=self.close_date_of_test)
        if remove_outliers:
            df_complete = remove_outlier_dataframe(
                df_complete, ticker+'_close')
        self.df_complete = df_complete
        self.df = self.df_complete.loc[self.open_date:self.close_date]

    @classmethod
    def create_configuration(self, df, num_of_scales: int = 3, period: int = 3, use_returns: bool = True):
        df = df.dropna()
        if use_returns:
            values = (df[df.columns[-1]]).values
            min_value = min(values)
            max_value = max(values)
        else:
            values = (df[df.columns[-1]])
            rol_mean = values.rolling(period).mean()
            rol_dev_pad = values.rolling(period).std()
            ratio = ((values-rol_mean)/rol_dev_pad).dropna()
            min_value, max_value = min(ratio), max(ratio)
        down_spacement = np.linspace(min_value, 0, num_of_scales+1)
        growth_spacement = np.linspace(0, max_value, num_of_scales+1)
        intervals = np.concatenate((down_spacement[:-1], growth_spacement))
        down, growth = [], []
        for scale in range(1, num_of_scales+1):
            down.append('d'+str(num_of_scales-scale+1))
            growth.append('g'+str(scale))
        states = down+growth
        configuration = []
        if use_returns:
            item_to_check = values
        else:
            item_to_check = ratio
        for value in item_to_check:
            index = bisect(intervals, value)
            if index > len(states):
                index -= 1
            if index < 0:
                index += 1
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
            df=self.df_complete, num_of_scales=self.num_of_scales, 
            period=self.period, use_returns=self.use_returns)
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
        config = self.create_configuration(
            df=self.df_complete, num_of_scales=self.num_of_scales, 
            period=self.period, use_returns=self.use_returns)
        return config[-1]
