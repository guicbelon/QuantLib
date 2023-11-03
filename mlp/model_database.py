from pykalman import KalmanFilter
from finta import TA
import pandas as pd
import numpy as np
import json
from datetime import timedelta, date, datetime
from ..database import Database
from ..constants import *
from ..data_management.data_cleaning import *
from ..markov import MarkovChain
from ..singleton import Singleton


class ModelDatabase(metaclass=Singleton):
    _TA_INFO = {
        'RSI': {'method': TA.RSI, 'upper_limit': 70, 'bottom_limit': 30},
        'MOM': {'method': TA.MOM, 'upper_limit': 0.5, 'bottom_limit': -0.5},
        'CCI': {'method': TA.CCI, 'upper_limit': 100, 'bottom_limit': -100},
        'STOCH': {'method': TA.STOCH, 'upper_limit': 80, 'bottom_limit': 20},
        'WILLIAMS': {'method': TA.WILLIAMS, 'upper_limit': -20, 'bottom_limit': -80},
    }

    def __init__(self,
                 open_date: str = None,
                 close_date: str = None,
                 rolling_periods: int = 3,
                 shift_data: bool = True,
                 remove_outliers_returns: bool = True) -> None:
        if close_date is None:
            close_date = date.today()
        close_date = pd.to_datetime(close_date)
        if open_date is None:
            open_date = close_date - timedelta(days=5*365)
        self.open_date = pd.to_datetime(open_date)
        self.close_date = close_date
        self.rolling_periods = rolling_periods
        self.shift_data = shift_data
        self.data = Database()
        self.base_df = self.data.get_info(
            ['USD/BRL',  'IBOV',  'SPX', 'DJI', 'CDI',
                'NASDAQ', 'IPCA', 'SELIC', 'IGPM'],
            open_date=open_date,
            close_date=close_date)
        self.json_file_name = 'files/df_data.json'
        self.ticker = None
        self.ticker_changed = True
        self.remove_outliers_returns = remove_outliers_returns
        self.close_date_to_train = self.close_date - timedelta(days=100)

    def set_ticker(self, ticker: str):
        self.ticker = ticker
        self.ticker_changed = True
        start_of_price = self.open_date - timedelta(days=60)
        price_info = self.data.get_info(
            ticker, start_of_price, self.close_date, 'ohlcv')
        price_info = price_info.dropna()
        self.ticker_info = price_info[[ticker+'_close']]
        future_info = price_info[[ticker+'_close']]
        self.returns_df = self.data.get_info(['RET_'+ticker],
                                             open_date=start_of_price,
                                             close_date=self.close_date).dropna()
        if self.shift_data:
            self.returns_df[f'RET_{ticker}_close'] = self.returns_df[f'RET_{ticker}_close'].shift(
                -1)
            self.returns_df = self.returns_df.fillna(0)
        if self.remove_outliers_returns:
            self.returns_df = remove_outlier_dataframe(
                df=self.returns_df, column=f'RET_{ticker}_close')
        self.first_prices = future_info[ticker+'_close'][:self.rolling_periods]
        config = MarkovChain.create_configuration(
            df = self.returns_df, num_of_scales=3)
        self.config_df = pd.DataFrame({'config': config}, index=self.returns_df.index)
        if self.rolling_periods is not None:
            future_info = future_info.rolling(
                window=self.rolling_periods).mean()
        self.future_info = future_info
        self.price_info = price_info.rename(
            columns={ticker+'_open': 'open', ticker + '_high': 'high',
                     ticker + '_low': 'low', ticker + '_close': 'close',
                     ticker + '_volume': 'volume'})

    def get_last_values(self, data, days_to_seek: int, upper_limit: float, bottom_limit: float):
        total_days = len(data)
        last_upper_values = (np.where(data > upper_limit))[0]
        if len(last_upper_values) == 0:
            upper_index = 0
        else:
            last_upper_values = total_days - last_upper_values[-1]
            upper_index = -1 + last_upper_values/days_to_seek
        if upper_index > 0:
            upper_index = 0
        last_bottom_values = (np.where(data < bottom_limit))[0]
        if len(last_bottom_values) == 0:
            bottom_index = 0
        else:
            last_bottom_values = total_days - last_bottom_values[-1]
            bottom_index = 1 - last_bottom_values/days_to_seek
        if bottom_index < 0:
            bottom_index = 0
        return bottom_index, upper_index

    def get_bigger_absolute(self, data: tuple):
        if abs(data[0]) > abs(data[1]):
            return data[0]
        return data[-1]

    def df_TA(self, indicator: str, periods=[2, 4, 5, 7, 10, 14, 20, 30], drop_index: bool = True):
        indicator = indicator.upper()
        method = self._TA_INFO[indicator]['method']
        if type(periods) is int:
            periods = [periods]
        df = self.price_info
        indexes = df.columns
        df = df.dropna()
        for period in periods:
            df[indicator+'_'+str(period)] = method(df, period)
        if drop_index:
            df = df.drop(columns=indexes)
        return df

    def indexes_TA(self, 
                   df_TA:pd.DataFrame, 
                   upper_limit: float, 
                   bottom_limit: float, 
                   days_to_seek: int = 30):
        df_TA = df_TA.dropna()
        dates = df_TA.index[days_to_seek:]
        periods = [col_name.split('_')[1] for col_name in df_TA.columns]
        ta_info = df_TA.columns[0].split('_')[0]
        new_df = pd.DataFrame(index=dates)
        for period in periods:
            TA_indexes = []
            for date in dates:
                TA_indexes_values = self.get_last_values(
                    df_TA[[ta_info+'_'+period]].loc[:date], 
                    days_to_seek, upper_limit, bottom_limit)
                TA_indexes.append(
                    self.get_bigger_absolute(TA_indexes_values))
            new_df['indexes_'+ta_info+'_'+period] = TA_indexes
        return new_df

    def df_MA(self, df,
              ma_type: str = 'SMA' or 'EMA',
              periods: [int] = [2, 4, 5, 7, 10, 14, 30, 45, 90],
              drop_index: bool = True):
        if type(periods) is int:
            periods = [periods]
        index = df.columns[-1]
        ma_type = ma_type.upper()
        df = df.dropna()
        if ma_type == 'SMA':
            method = df[index].rolling
        else:
            method = df[index].ewm
        for period in periods:
            df[ma_type+'_'+str(period) + '_' + index] = method(period).mean()
        if drop_index:
            df = df.drop(columns=index)
        return df

    def df_kalman(self, df,
                  covariances: [float] = [0.005, 0.01, 0.03, 0.05],
                  drop_index: bool = True):
        if type(covariances) is float:
            covariances = [covariances]
        index = df.columns[-1]
        df = df.dropna()
        initial = df[index][0]
        df_to_filter = df[[index]]
        for covariance in covariances:
            try:
                kf = KalmanFilter(transition_matrices=[1],
                                  observation_matrices=[1],
                                  initial_state_mean=initial,
                                  initial_state_covariance=1,
                                  observation_covariance=1,
                                  transition_covariance=covariance)
                state_means, _ = kf.filter(df_to_filter)
                df['KLM_'+str(covariance)+'_'+index] = state_means
            except:
                pass
        if drop_index:
            df = df.drop(columns=index)
        return df

    def create_moving_info(self, df, drop_index: bool = False):
        sma_df = self.df_MA(df, 'SMA', drop_index=drop_index)
        ema_df = self.df_MA(df, 'EMA', drop_index=True)
        kalman_df = self.df_kalman(df, drop_index=True)
        return pd.concat([sma_df, ema_df, kalman_df], axis=1)

    def get_best_correlated(self, df_reference, df, ignore_reference: bool = True):
        df_to_check = pd.concat([df_reference, df], axis=1)
        df_corr = df_to_check.corr()
        index_reference = df_reference.columns[0]
        best_correlated = df_corr[index_reference].sort_values(
            ascending=False).index[1:][0]
        if ignore_reference:
            return df_to_check[[best_correlated]]
        return df_to_check[[index_reference, best_correlated]]

    def create_TA_database(self, ignore_index: bool = False, days_to_seek: int = 30,):
        dfs_to_concat = []
        if not ignore_index:
            dfs_to_concat.append(self.future_info)
        for ta_indicator in self._TA_INFO.keys():
            raw_ta_indic = self.df_TA(ta_indicator, drop_index=True)
            ta_indic = self.get_best_correlated(self.future_info, raw_ta_indic)
            dfs_to_concat.append(ta_indic)
        df = pd.concat(dfs_to_concat, axis=1)
        return df

    def create_vol_database(self, periods: [int] = [2, 3, 5, 7, 14, 20, 30, 60, 90, 252]):
        if type(periods) is int:
            periods = [periods]
        vols_to_fetch = []
        for period in periods:
            vols_to_fetch.append('VOL'+str(period)+'_'+self.ticker)
        df = self.data.get_info(vols_to_fetch, self.open_date, self.close_date)
        return df

    def create_volume_database(self):
        volume_df = self.data.get_info(
            self.ticker, open_date=self.open_date, close_date=self.close_date, info='volume')
        moving_info = self.create_moving_info(volume_df)
        best_df = self.get_best_correlated(self.future_info, moving_info)
        return best_df

    def get_windowed_df(self, df, previous_days: int = 3):
        df = df.dropna()
        date_range = df.index
        close_date = date_range[-1]
        dates = []
        X = []
        last_time = False
        index = df.columns[-1]
        date_index = 3
        target_date = date_range[date_index]
        while True:
            df_subset = df.loc[:target_date].tail(previous_days+1)
            if len(df_subset) != previous_days+1:
                raise Exception(
                    f'Error: Window of size {previous_days} is too large for date {target_date}')
            values = df_subset[index].to_numpy()
            x = values[:-1]
            dates.append(target_date)
            X.append(x)
            date_index += 1
            next_date = date_range[date_index]
            if last_time:
                break
            target_date = next_date
            if target_date == close_date:
                date_index -= 1
                last_time = True
        ret_df = pd.DataFrame({})
        ret_df['date'] = dates
        X = np.array(X)
        for i in range(0, previous_days):
            X[:, i]
            ret_df[f'd-{previous_days-i}'] = X[:, i]
        ret_df.set_index('date', inplace=True)
        return ret_df

    def svd_orthogonalization(self, df):
        u, s, vh = np.linalg.svd(df, full_matrices=False)
        orthogonalized_df = np.dot(u, np.diag(s))
        return pd.DataFrame(orthogonalized_df, columns=df.columns)

    def price_ratio_df(self, period=3):
        df = self.future_info
        values = (df[df.columns[-1]])
        rol_mean = values.rolling(period).mean()
        rol_dev_pad = values.rolling(period).std()
        ratio = ((values-rol_mean)/rol_dev_pad).dropna()
        df_ratio = pd.DataFrame({'ratio': ratio.values}, index=ratio.index)
        df_ratio = df_ratio.rename(
            columns={df.columns[-1]: 'ratio'})
        moving_info = self.create_moving_info(df_ratio)
        best_info = self.get_best_correlated(self.future_info, moving_info)
        return best_info

    def create_specific_base_df(self):
        dfs_to_concat = []
        base_df = self.base_df.dropna()
        for index in base_df.columns:
            df_to_check = base_df[[index]]
            moving_info = self.create_moving_info(df_to_check)
            best_info = self.get_best_correlated(self.future_info, moving_info)
            dfs_to_concat.append(best_info)
        ta_df = self.create_TA_database(ignore_index=True)
        dfs_to_concat.append(ta_df)
        df = pd.concat(dfs_to_concat, axis=1)
        return df

    def create_seazonal_df(self, dates_index):
        last_date_year = dates_index[-1].year
        week_days = pd.get_dummies([date.weekday() for date in dates_index], columns=[
                                   str(i) for i in range(5)])
        months = pd.get_dummies([date.month for date in dates_index], columns=[
                                str(i) for i in range(2, 1)])
        years = pd.get_dummies([date.year for date in dates_index], columns=[
                               str(i) for i in range(last_date_year-5, last_date_year+1)])
        seasonal_df = pd.concat([week_days, months, years], axis=1)
        return seasonal_df

    def create_default_database(self):
        price_ratio_df = self.price_ratio_df()
        volume_df = self.create_volume_database()
        vol_df = self.data.get_info(
            'VOL3_'+self.ticker, self.open_date, self.close_date)
        macd = TA.MACD(self.price_info, 12, 26, 9)
        bbands = TA.BBANDS(self.price_info).dropna()
        bbands['BBANDS_delta'] = bbands['BB_UPPER'] - bbands['BB_LOWER']
        bbands = bbands[['BBANDS_delta']]
        default_database = pd.concat([volume_df, vol_df,
                                      price_ratio_df, macd, bbands], axis=1)
        return default_database
    
    def create_equal_distributed_database(self, df:pd.DataFrame, column_name:str):
        counts = df[[column_name]].value_counts()
        min_count = counts.min()
        date_range = df.index
        df_column_data = df[[column_name]].values.reshape(-1)
        possible_states = list(set(df_column_data.tolist()))
        states_indexes = []
        for state in possible_states:
            indexes = np.where(df_column_data == state)
            states_indexes.append((indexes[0]))
        indexes_to_choose = np.array([])
        for states_index in states_indexes:
            indexes_to_choose = np.concatenate(
                (indexes_to_choose, np.random.choice(states_index, min_count, replace=False)))
        indexes_to_choose = indexes_to_choose.astype(int)
        indexes_to_choose = np.random.choice(
            indexes_to_choose, size=indexes_to_choose.shape[0], replace=False)
        dates_to_choose = date_range[indexes_to_choose]
        return df.loc[dates_to_choose]

    def create_training_database(self):
        if self.ticker is None:
            raise Exception('No ticker was given')
        default_database = self.create_default_database()
        specific_database = self.create_specific_base_df()
        df = pd.concat([self.price_info,self.returns_df,self.config_df, default_database,
                       specific_database], axis=1)
        df = df.loc[:self.close_date_to_train]
        df = df.dropna()
        if len(df) == 0:
            raise Exception('Problems in creating database.')
        col_names = specific_database.columns
        with open(self.json_file_name, 'r') as json_file:
            loaded_data = json.load(json_file)
        col_data = list(col_names)
        str_date = self.close_date.strftime("%Y-%m-%d")
        loaded_data[self.ticker] = {'col': col_data, 'date': str_date}
        with open(self.json_file_name, 'w') as json_file:
            json.dump(loaded_data, json_file)
        return df

    def create_test_database(self):
        moving_info = set(['EMA', 'SMA', 'KLM'])
        with open(self.json_file_name, 'r') as json_file:
            loaded_data = json.load(json_file)
        ticker_info = loaded_data[self.ticker]
        cols_info = ticker_info['col']
        info_splits = []
        dfs_to_concat = []
        for info in cols_info:
            splits = list(info.split('_'))
            if splits[-1] == 'close':
                splits = splits[:-1]
            info_splits.append(splits)
        for info in info_splits:
            if len(info) == 1:
                df = self.data.get_info(
                    info[0], self.open_date, self.close_date)
                dfs_to_concat.append(df)
            else:
                if info[0] == 'RET':
                    df = self.data.get_info(
                        'RET_'+info[1], self.open_date, self.close_date)
                    dfs_to_concat.append(df)
                elif info[0] in moving_info:
                    ticker_to_fetch = info[2]
                    if len(info) == 4:
                        ticker_to_fetch = info[2]+'_'+info[3]
                    base_df = self.data.get_info(
                        ticker_to_fetch, self.open_date, self.close_date)
                    if info[0] == 'EMA' or info[0] == 'SMA':
                        df = self.df_MA(
                            df=base_df, ma_type=info[0], periods=int(info[1]))
                    elif info[0] == 'KLM':
                        df = self.df_kalman(
                            df=base_df, covariances=float(info[1]))
                    dfs_to_concat.append(df)
                elif info[0] in self._TA_INFO.keys():
                    df = self.df_TA(indicator=info[0], periods=int(info[1]))
                    dfs_to_concat.append(df)
                elif info[0] == 'indexes':
                    pass
                else:
                    ticker_data = info[0]
                    for info_data in info[1:]:
                        ticker_data += '_'+info_data
                    raise Exception(
                        f'Uncapable of creating {ticker_data} information!')
        specific_database = pd.concat(dfs_to_concat, axis=1)
        default_database = self.create_default_database()
        test_df = pd.concat(
            [self.price_info,self.returns_df,self.config_df, default_database, specific_database], axis=1)
        test_df = test_df.dropna()
        test_df = test_df.loc[self.close_date_to_train +
                              timedelta(days=1):self.close_date]
        if len(test_df) == 0:
            raise Exception('Problems in creating database.')
        return test_df

    def get_training_and_test_database(self):
        if self.ticker_changed:
            self.created_train_database = self.create_training_database()
            self.created_test_database = self.create_test_database()
            self.ticker_changed = False
        data = {
            'train': self.created_train_database,
            'test': self.created_test_database
        }
        return data

    def recreate_database(self, df_predicted):
        df_original = self.future_info
        col_name = df_original.columns[0]
        max_date_original = df_predicted.index[0]
        self.first_date_predicted = df_predicted.index[1]
        df_original = df_original.loc[:max_date_original]
        df_original = df_original.rename(columns={col_name: 'predict'})
        df_predicted = df_predicted.loc[self.first_date_predicted:]
        df = pd.concat([df_original, df_predicted])
        return df

    def recreate_prices(self, recreated_database):
        indexes = recreated_database.index
        recreated_database = recreated_database.dropna()
        periods = 3
        prices = list(self.first_prices.values)
        for mean_value in recreated_database.values[1:]:
            last_sum = sum(prices[-(periods-1):])
            new_price = mean_value[0]*periods - last_sum
            prices.append(new_price)
        recreated_df = pd.DataFrame({'prices': prices}, index=indexes)
        return recreated_df

    def get_states_returns(self, recreated_database):
        data = recreated_database.dropna()
        daily_retuns = (data-data.shift(1))/data.shift(1)
        date_to_ignore = self.first_date_predicted
        daily_retuns = daily_retuns.drop(date_to_ignore)
        daily_retuns = daily_retuns.dropna()
        daily_retuns = remove_outlier_dataframe(daily_retuns, 'predict')
        mk = MarkovChain(self.ticker, self.close_date, use_returns=True)
        df_states = mk.create_configuration(daily_retuns)
        df_states = pd.DataFrame(
            {'states': df_states}, index=daily_retuns.index)
        df_states = df_states.loc[self.first_date_predicted:]
        return df_states
