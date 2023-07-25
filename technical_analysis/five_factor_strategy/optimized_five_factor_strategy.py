import pandas as pd
import numpy as np
from datetime import date, timedelta
from matplotlib import pyplot as plt
from finta import TA
from ...database import Database
from ...backtest.trades import Trade
from .five_factor_optimizer import FiveFactorOptimizer
from .five_factor_strategy import FiveFactorStrategy


class OptmizedFiveFactorStrategy:
    def __init__(
        self,
        traded_ticker: str,
        open_date: str,
        close_date: str = None,
        stop_loss: float = None,
        profit_margin: float = None,
        check_ticker: str = None,
        previous_years_to_check: int = 4
    ) -> None:
        self.traded_ticker = traded_ticker
        self.open_date = pd.to_datetime(open_date)
        self.close_date = pd.to_datetime(close_date)
        if close_date == None:
            close_date = pd.to_datetime(date.today()) -timedelta(days=1)
        self.stop_loss = stop_loss
        self.profit_margin = profit_margin
        self.check_ticker = check_ticker
        if check_ticker == None:
            self.check_ticker = traded_ticker
        start_date_to_check = self.open_date - \
            timedelta(weeks=52*previous_years_to_check)

        optmizer = FiveFactorOptimizer(
            traded_ticker=self.traded_ticker, open_date=start_date_to_check,
            close_date=self.open_date, stop_loss=self.stop_loss,
            profit_margin=self.profit_margin, check_ticker=self.check_ticker)
        self.optmizer_info = optmizer.strategy_info()
        self.cci_period = self.optmizer_info['cci_period']
        self.williams_period = self.optmizer_info['williams_period']
        self.stoch_period = self.optmizer_info['stoch_period']
        self.rsi_period = self.optmizer_info['rsi_period']
        self.momentum_period = self.optmizer_info['momentum_period']

        self.strategy = FiveFactorStrategy(
            traded_ticker=self.traded_ticker, open_date=self.open_date,
            close_date=self.close_date, stop_loss=self.stop_loss,
            profit_margin=self.profit_margin, check_ticker=self.check_ticker,
            cci_period=self.cci_period, williams_period=self.williams_period,
            stoch_period=self.stoch_period, rsi_period=self.rsi_period, 
            momentum_period=self.momentum_period)

    @property
    def optmized_info(self):
        return self.optmizer_info

    def intersection_data(self):
        return self.strategy.intersection_info_data

    def get_trades(self):
        return self.strategy.get_trades()

    def plot(self, start=None, close=None):
        self.strategy.plot(start=start,close=close)

