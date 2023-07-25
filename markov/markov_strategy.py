from quantitiative_finances.portfolio_management.dendogram_manager import DendogramPortfolio
from quantitiative_finances.portfolio_management.IBOV_related import ibov_related
from quantitiative_finances import Database
from ..constants import *
from ..trader import Trader
from datetime import date, timedelta
from .markov_chain import MarkovChain
import pandas as pd

class MarkovStrategy:
    def __init__(self, 
                 start_of_train: int = None, 
                 date_to_check: str = None, 
                 only_buy: bool = True
            ) -> None:
        self.trader = Trader()
        self.tickers = self.trader.get_tickers()
        if date_to_check is None:
            date_to_check = date.today()
        self.date_to_check = pd.to_datetime(date_to_check)
        self.date_to_check_minus_one = self.date_to_check - timedelta(days=1)
        if start_of_train is None:
            start_of_train = self.date_to_check - timedelta(days=10*365)
            self.only_buy = only_buy
        self.start_train = start_of_train
        self.tickers_to_trade = None
        self.errors = None

    def _get_all_available_to_trade(self):
        if self.tickers_to_trade is not None:
            return self.tickers_to_trade
        error = 0
        tickers_to_trade = {}
        for ticker in self.tickers:
            try:
                mkv = MarkovChain(ticker=ticker, open_date=self.start_train,
                                  close_date=self.date_to_check_minus_one,
                                  close_date_of_test=self.date_to_check)
                states_to_trade = mkv.get_states_to_trade()
                current_state = mkv.get_current_state()
                if current_state in states_to_trade:
                    if current_state[0] == 'g':
                        tickers_to_trade[ticker] = 'buy'
                    else:
                        if not self.only_buy:
                            tickers_to_trade[ticker] = 'sell'
            except:
                error += 1
                pass
        self.errors = error
        self.tickers_to_trade = tickers_to_trade
        return self.tickers_to_trade

    def initialize(self, operate:bool=True):
        tickers_to_trade = self._get_all_available_to_trade()
        ibov_unrelated=[]
        for ticker in tickers_to_trade.keys():
            is_related = ibov_related(ticker,start_date=self.start_train, end_date=self.date_to_check)
            if not is_related:
                ibov_unrelated.append(ticker)
        tickers_to_operate = ibov_unrelated
        if len(tickers_to_operate)>AMOUNT_TO_OPERATE:
            portfolio_manager = DendogramPortfolio(
                tickers_to_operate, self.start_train, self.date_to_check)
            tickers_to_operate = portfolio_manager.tickers_to_operate
        print(tickers_to_operate)
        if operate:
            self.trader.close_all_orders()
            for ticker in tickers_to_operate:
                try:
                    self.trader.alocate(ticker=ticker, num_of_tickers=len(
                        tickers_to_operate),operation= tickers_to_trade[ticker],
                        comment='Markov Strategy')
                except:
                    pass
        Database().reset()