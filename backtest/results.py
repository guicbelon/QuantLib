import pandas as pd
import numpy as np
from datetime import date
from matplotlib import pyplot as plt
from finta import TA

class Results:
    def __init__(self, trades: list) -> None:
        self.trades = trades
        self._preprocessed = False
        self.number_of_trades = len(self.trades)

    def _preprocess(self):
        if self._preprocessed:
            return
        daily_returns_values = pd.DataFrame()
        self.pnl_values=[]
        win_pnl = 0
        
        for trade in self.trades:
            daily_returns_values = pd.concat(
                [daily_returns_values, trade.daily_returns()], axis=0)     
            pnl = trade.PnL()
            self.pnl_values.append(pnl)
            if pnl>0:
                win_pnl +=1
        
        if self.number_of_trades ==0:
            self.average_pnl_value = 0
            self.hit_ratio_value = 0
        else:
            self.total_pnl_value = sum(self.pnl_values)
            self.average_pnl_value = self.total_pnl_value/self.number_of_trades
            self.hit_ratio_value = win_pnl/self.number_of_trades

        daily_returns_values.groupby(pd.Grouper(freq='1D')).mean()
        self.cumulated_return_values = (
            daily_returns_values.fillna(0)+1).cumprod()-1
        
        self.daily_returns_values= daily_returns_values
        self._preprocessed = True

    def returns(self):
        self._preprocess()
        return self.daily_returns_values
    
    def total_pnl(self):
        self._preprocess()
        return self.total_pnl_value

    def cumulated_return(self):
        self._preprocess()
        return self.cumulated_return_values
    
    def hit_ratio(self):
        self._preprocess()
        return self.hit_ratio_value
    
    def average_pnl(self):
        self._preprocess()
        return self.average_pnl_value
    
    def hit_multiplier(self):
        self._preprocess()
        return((self.hit_ratio()**2)*(self.number_of_trades**(1/2)))
    
    def results_info(self):
        dict_info={
            'hit_ratio': self.hit_ratio(),
            'average_pnl': self.average_pnl(),
            'cumulated_return':self.cumulated_return()[-1],
            'total_pnl':self.total_pnl(),
            'hit_multiplier':self.hit_multiplier()
        }
        return dict_info
    
    def plot(self,start=None,end=None):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        plt.style.use('seaborn')
        returns = self.cumulated_return().loc[start:end]
        plt.gcf().set_size_inches(16, 8)
        plt.plot(returns)
        plt.show()



