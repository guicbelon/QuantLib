import pandas as pd
import numpy as np
from datetime import date
from ...database import Database
from .five_factor_strategy import FiveFactorStrategy
from ...backtest.results import Results

    
class FiveFactorOptimizer:
    def __init__(
        self,
        traded_ticker: str,
        open_date: str,
        close_date: str = None,
        stop_loss: float = None,
        profit_margin: float = None,
        check_ticker: str = None,
        begin: int=5,
        end:int=51,
        step:int=2
    ) -> None:
        self.traded_ticker = traded_ticker
        self.open_date = open_date
        self.close_date = close_date
        if close_date == None:
            close_date = pd.to_datetime(date.today())
        self.stop_loss = stop_loss
        self.profit_margin = profit_margin
        self.check_ticker = check_ticker
        if check_ticker == None:
            self.check_ticker = traded_ticker

        self.best_cci_period = 14
        self.best_williams_period = 14
        self.best_stoch_period = 14
        self.best_rsi_period = 14
        self.best_momentum_period = 14
        self.best_value_of_param = -np.inf
        self.begin=begin
        self.end=end
        self.step=step
        self._preprocessed = False

    def _preprocess(self):
        if self._preprocessed == True:
            return
        last_time_value_of_param = self.best_value_of_param
        for time in range(10):
            for cci_period in range(self.begin,self.end,self.step):
                strategy = FiveFactorStrategy(
                    traded_ticker=self.traded_ticker, open_date=self.open_date,
                    close_date=self.close_date, profit_margin=self.profit_margin,
                    stop_loss=self.stop_loss, check_ticker=self.check_ticker,
                    cci_period=cci_period, williams_period=self.best_williams_period,
                    stoch_period=self.best_stoch_period, rsi_period=self.best_rsi_period,
                    momentum_period=self.best_momentum_period
                )
                results = Results(strategy.get_trades())
                value_of_param = results.total_pnl()
                if value_of_param > self.best_value_of_param:
                    self.best_value_of_param = value_of_param
                    self.best_cci_period = cci_period

            for williams_period in range(self.begin,self.end,self.step):
                strategy = FiveFactorStrategy(
                    traded_ticker=self.traded_ticker, open_date=self.open_date,
                    close_date=self.close_date, profit_margin=self.profit_margin,
                    stop_loss=self.stop_loss, check_ticker=self.check_ticker,
                    cci_period=self.best_cci_period, williams_period=williams_period,
                    stoch_period=self.best_stoch_period, rsi_period=self.best_rsi_period,
                    momentum_period=self.best_momentum_period
                )
                results = Results(strategy.get_trades())
                value_of_param = results.total_pnl()
                if value_of_param > self.best_value_of_param:
                    self.best_value_of_param = value_of_param
                    self.best_williams_period = williams_period

            for stoch_period in range(self.begin,self.end,self.step):
                strategy = FiveFactorStrategy(
                    traded_ticker=self.traded_ticker, open_date=self.open_date,
                    close_date=self.close_date, profit_margin=self.profit_margin,
                    stop_loss=self.stop_loss, check_ticker=self.check_ticker,
                    cci_period=self.best_cci_period, williams_period=self.best_williams_period,
                    stoch_period=stoch_period, rsi_period=self.best_rsi_period,
                    momentum_period=self.best_momentum_period
                )
                results = Results(strategy.get_trades())
                value_of_param = results.total_pnl()
                if value_of_param > self.best_value_of_param:
                    self.best_value_of_param = value_of_param
                    self.best_stoch_period = stoch_period

            for rsi_period in range(self.begin,self.end,self.step):
                strategy = FiveFactorStrategy(
                    traded_ticker=self.traded_ticker, open_date=self.open_date,
                    close_date=self.close_date, profit_margin=self.profit_margin,
                    stop_loss=self.stop_loss, check_ticker=self.check_ticker,
                    cci_period=self.best_cci_period, williams_period=self.best_williams_period,
                    stoch_period=self.best_stoch_period, rsi_period=rsi_period,
                    momentum_period=self.best_momentum_period
                )
                results = Results(strategy.get_trades())
                value_of_param = results.total_pnl()
                if value_of_param > self.best_value_of_param:
                    self.best_value_of_param = value_of_param
                    self.best_rsi_period = rsi_period

            for momentum_period in range(self.begin,self.end,self.step):
                strategy = FiveFactorStrategy(
                    traded_ticker=self.traded_ticker, open_date=self.open_date,
                    close_date=self.close_date, profit_margin=self.profit_margin,
                    stop_loss=self.stop_loss, check_ticker=self.check_ticker,
                    cci_period=self.best_cci_period, williams_period=self.best_williams_period,
                    stoch_period=self.best_stoch_period, rsi_period=self.best_rsi_period,
                    momentum_period=momentum_period
                )
                results = Results(strategy.get_trades())
                value_of_param = results.total_pnl()
                if value_of_param > self.best_value_of_param:
                    self.best_value_of_param = value_of_param
                    self.best_momentum_period = momentum_period
            if last_time_value_of_param == self.best_value_of_param:
                break
            last_time_value_of_param = self.best_value_of_param
        self._preprocessed = True

    def strategy_info(self):
        self._preprocess()
        info_dct = {
            'traded_ticker': self.traded_ticker,
            'cci_period':self.best_cci_period, 
            'williams_period':self.best_williams_period,
            'stoch_period':self.best_stoch_period, 
            'rsi_period':self.best_rsi_period, 
            'momentum_period':self.best_momentum_period
        }
        return info_dct
    
    def best_param(self):
        self._preprocess()
        return self.best_value_of_param

