from ..database import Database
from ..constants import *
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)

class DendogramPortfolio:
    def __init__(self, 
                 tickers:list, 
                 start_date:str=None, 
                 close_date:str=None, 
                 tickers_to_select:int=None
            ) -> None:
        self.tickers = tickers
        if tickers_to_select is None:
            tickers_to_select = AMOUNT_TO_OPERATE
        if tickers_to_select>len(tickers):
            tickers_to_select = len(tickers)
        self.tickers_to_select = tickers_to_select
        self.df = Database().get_info(tickers, start_date, close_date)
        self.cov = self.df.cov
        self.corr = self.df.corr()
        self.dist = ((1-self.corr)/2.)**5
        self.link = sch.linkage(self.dist,'single')
        self.nodes = []
        for row in self.link[::-1]:
            left_number = int(row[0])
            right_number = int(row[1])
            try:
                ticker = tickers[left_number]
                self.nodes.append(ticker)
            except IndexError:
                pass
            try:
                ticker = tickers[right_number]
                self.nodes.append(ticker)
            except IndexError:
                pass   
        indexes = np.linspace(0, len(self.nodes)-1, self.tickers_to_select, dtype=int)
        self._tickers_to_operate = [self.nodes[i] for i in indexes]
        
    @property
    def tickers_to_operate(self):
        return self._tickers_to_operate

    def display_dendogram(self):
        sch.dendrogram(self.link)
        plt.show()
    
    def other_tickers_to_operate(self,amount:int):
        indexes = np.linspace(0, len(self.nodes)-1, amount, dtype=int)
        new_tickers_to_operate = [self.nodes[i] for i in indexes]
        return new_tickers_to_operate


