import pandas as pd
import time
import numpy as np
import json
from ..portfolio_management.dendogram_manager import DendogramPortfolio
from ..portfolio_management.IBOV_related import ibov_related
from ..database import Database
from ..constants import *
from ..trader import Trader
from datetime import date, timedelta
from .mlp_model import MLPModel

class MLPStrategy:
    def __init__(self, open_date:str=None, close_date:str=None):
        if close_date is None:
            close_date = date.today()
        close_date = pd.to_datetime(close_date)
        if open_date is None:
            open_date = close_date - timedelta(days=5*365)
        self.open_date = pd.to_datetime(open_date)
        self.close_date = close_date
        self.trader = Trader()
        self.tickers = self.trader.get_tickers()
        self.model = MLPModel(open_date=open_date, close_date=close_date)
        self.json_file_name = 'files/df_data.json'
        with open(self.json_file_name, 'r') as json_file:
            loaded_data = json.load(json_file)
        self.trained_tickers = sorted(list(loaded_data.keys()))
        self.all_tickers = sorted(self.trader.all_tickers)

    def train_all_tickers(self):
        for ticker in self.all_tickers:
            if ticker not in self.trained_tickers:
                try:
                    self.model.set_ticker(ticker)
                    self.model.train()
                    print(ticker, ' trained!')
                except:
                    pass
    
    def update_all_tickers(self):
        for ticker in self.all_tickers:
            try:
                self.model.set_ticker(ticker)
                self.model.train()
                print(ticker, ' trained!')
            except:
                pass
    
    def update_trained_tickers(self):
        for ticker in self.trained_tickers:
            try:
                self.model.set_ticker(ticker)
                self.model.train()
                print(ticker, ' trained!')
            except:
                pass
    
    def _get_all_available_to_trade(self):
        mlp_model = MLPModel(self.open_date, self.close_date)
        available_to_trade = set(self.tickers)
        tickers_to_trade={}
        for ticker in self.trained_tickers:
            if ticker in available_to_trade:
                try:
                    mlp_model.set_ticker(ticker)
                    current_state = mlp_model.get_current_state()
                    if current_state=='g2' or current_state=='g3':
                        tickers_to_trade[ticker] = current_state
                except:
                    pass
        return tickers_to_trade
        
    def initialize(self, operate:bool=True):
        tickers_to_operate = self._get_all_available_to_trade()
        if len(tickers_to_operate.keys())>AMOUNT_TO_OPERATE:
            portfolio_manager = DendogramPortfolio(
                tickers_to_operate, self.start_train, self.date_to_check)
            tickers_to_operate = portfolio_manager.tickers_to_operate
        print(tickers_to_operate)
        if operate:
            self.trader.close_all_orders()
            for ticker in tickers_to_operate:
                try:
                    self.trader.alocate(ticker=ticker, num_of_tickers=len(
                        tickers_to_operate),operation= 'buy',
                        comment='MLP Strategy')
                except:
                    pass
        Database().reset()

