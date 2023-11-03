import pandas as pd
import numpy as np
from datetime import date, timedelta
from matplotlib import pyplot as plt
from finta import TA
from ...database import Database
from ...backtest.trades import Trade


class FiveFactorStrategy:
    def __init__(
        self,
        traded_ticker: str,
        open_date: str,
        close_date: str = None,
        stop_loss: float = None,
        profit_margin: float = None,
        check_ticker: str = None,
        cci_period: int = 14,
        williams_period: int = 14,
        stoch_period: int = 14,
        rsi_period: int = 14,
        momentum_period: int = 14
    ) -> None:
        self.traded_ticker = traded_ticker
        self.open_date = open_date
        self.close_date = close_date
        if close_date == None:
            close_date = pd.to_datetime(date.today()) -timedelta(days=1)
        self.stop_loss = stop_loss
        self.profit_margin = profit_margin
        self.check_ticker = check_ticker
        if check_ticker == None:
            self.check_ticker = traded_ticker
        self.cci_period = cci_period
        self.williams_period = williams_period
        self.stoch_period = stoch_period
        self.rsi_period = rsi_period
        self.momentum_period = momentum_period
        open_date_to_check = pd.to_datetime(open_date) - timedelta(days=55)
        self.check_ticker_assets = Database().get_info(
            tickers=self.check_ticker, open_date=open_date_to_check, close_date=close_date, info='ohlcv')
        self.check_ticker_assets = self.check_ticker_assets.rename(
            columns={self.check_ticker+'_open': 'open',
                     self.check_ticker + '_high': 'high',
                     self.check_ticker + '_low': 'low',
                     self.check_ticker + '_close': 'close',
                     self.check_ticker + '_volume': 'volume'
                     })
        self.traded_ticker_assets = Database().get_info(
            tickers=self.traded_ticker, open_date=open_date,
            close_date=close_date)[[traded_ticker+'_close']]
        self.data_created = False
        self.trades_found = False

    def check_intersection(self, series, intersect_value):
        intersect = [None]*len(series)
        for i in range(1, len(series)):
            if series[i] < intersect_value and series[i-1] > intersect_value:
                intersect[i] = -1  # intersect from top
            if series[i] > intersect_value and series[i-1] < intersect_value:
                intersect[i] = 1  # intersect from bottom
        return intersect

    def cci(self, df, period=14):
        cci = list(TA.CCI(df, period=period))
        intersect_cci = self.check_intersection(cci, 0)
        df = pd.DataFrame(intersect_cci, columns=[
            'cci_intersection'])
        return df

    def williams_r(self, df, period=14):
        williams = list(TA.WILLIAMS(df, period=period))
        intesect_williams = self.check_intersection(williams, -50)
        df = pd.DataFrame(intesect_williams, columns=[
            'williams_intersection'])
        return df

    def stoch_k(self, df, period=14):
        stoch = list(TA.STOCH(df, period=period))
        intersect_stoch = self.check_intersection(stoch, 50)
        df = pd.DataFrame(intersect_stoch, columns=[
            'stoch_intersection'])
        return df

    def rsi(self, df, period=14):
        rsi = list(TA.RSI(df, period=period))
        intersect_rsi = self.check_intersection(rsi, 50)
        df = pd.DataFrame(intersect_rsi, columns=[
            'rsi_intersection'])
        return df

    def momentum(self, df, period=14):
        mom = list(TA.MOM(df, period=period))
        intersect_mom = self.check_intersection(mom, 0)
        df = pd.DataFrame(intersect_mom, columns=[
            'momentum_intersection'])
        return df

    def intersection_data(self):
        if self.data_created:
            return self.intersection_info_data
        dates = self.check_ticker_assets.index
        df = pd.DataFrame()
        df = pd.concat([df, self.cci(self.check_ticker_assets,
                                     self.cci_period)], axis=1)
        df = pd.concat([df, self.williams_r(self.check_ticker_assets,
                                            self.williams_period)], axis=1)
        df = pd.concat([df, self.stoch_k(self.check_ticker_assets,
                                         self.stoch_period)], axis=1)
        df = pd.concat([df, self.rsi(self.check_ticker_assets,
                                     self.rsi_period)], axis=1)
        df = pd.concat([df, self.momentum(self.check_ticker_assets,
                                          self.momentum_period)], axis=1)
        df = pd.concat([pd.DataFrame({'dates': dates}), df], axis=1)
        df = df.set_index('dates')
        self.intersection_info_data = df.loc[self.open_date:self.close_date]
        self.data_created = True
        return self.intersection_info_data

    def get_trades(self):
        if self.trades_found:
            return self.trades
        buy_signal = []
        sell_signal = []
        dates_of_trade = []
        self.trades = []
        intersection = self.intersection_data()
        dates = intersection.index
        open_date = None
        buy_reason=None
        sell_reason =None

        for index in range(1, len(dates)-1):
            thereis1 = False
            thereis_1 = False
            prev_values = list(intersection.loc[dates[index-1]].values)
            values = list(intersection.loc[dates[index]].values)
            next_values = list(intersection.loc[dates[index+1]].values)

            for value in range(len(values)):
                if (values[value] == -1 or prev_values[value] == -1 or next_values[value] == -1):
                    thereis_1 = True
                if (values[value] == 1) and thereis_1 == False:
                    buy_reason=value

                if (values[value] == 1 or prev_values[value] == 1 or next_values[value] == 1):
                    thereis1 = True
                if (values[value] == -1) and thereis1 == False:
                    sell_reason=value

            if buy_reason is not None and thereis_1 == False and open_date == None and index != len(dates)-1:
                buy_value = self.traded_ticker_assets.values[index]
                dates_of_trade.append(dates[index])
                buy_signal.append(buy_value)
                sell_signal.append(None)
                open_date = dates[index+1]

            elif ((sell_reason == buy_reason and thereis1 == False) or index == len(dates)-1) and open_date is not None:
                sell_value = self.traded_ticker_assets.values[index]
                sell_signal.append(sell_value)
                buy_signal.append(None)
                trade = Trade(ticker=self.traded_ticker, open_date=open_date,
                              close_date=dates[index +1], 
                              stop_loss=self.stop_loss,
                              profit_margin=self.profit_margin, 
                              exposition='long')
                dates_of_trade.append(trade.last_date)
                self.trades.append(trade)
                open_date = None
                buy_reason = None
                sell_reason = None
        dct_to_plot = {
            'dates': dates_of_trade,
            'buy': buy_signal,
            'sell': sell_signal
        }
        self.df_to_plot = pd.DataFrame(dct_to_plot)
        self.df_to_plot = self.df_to_plot.set_index('dates')

        self.trades_found = True
        return self.trades

    def plot(self, start=None, close=None):
        self.get_trades()
        begin = self.open_date
        end = self.close_date
        if start is not None:
            begin = pd.to_datetime(start)
        if close is not None:
            end = pd.to_datetime(close)
        trade_df = self.df_to_plot.loc[begin:end]
        ticker_df = self.traded_ticker_assets.loc[begin:end]
        plt.style.use('seaborn')
        plt.plot(ticker_df[self.traded_ticker+'_close'],
                 label=self.traded_ticker, color='blue')
        plt.plot(trade_df['buy'], marker='^', color='green', markersize=9)
        plt.plot(trade_df['sell'], marker='v', color='red', markersize=9)
        plt.gcf().set_size_inches(16, 8)
        plt.legend()
        plt.show()
