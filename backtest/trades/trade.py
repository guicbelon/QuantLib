import pandas as pd
import numpy as np
from datetime import date
from ...constants import *
from ...database import Database

class Trade:
    def __init__(
            self,
            ticker: str,
            open_date: str,
            close_date: str = None,
            stop_loss: float = None,
            profit_margin: float = None,
            exposition: str = 'long',
            spread_cost: float = DEFAULT_SPREAD_COST,
            trading_cost: float = DEFAULT_TRADING_COST
    ) -> None:
        self.ticker = ticker
        self.open_date = pd.to_datetime(open_date)
        self.close_date = pd.to_datetime(close_date)
        self.stop_loss = stop_loss
        self.profit_margin = profit_margin
        self.exposition = exposition
        self.cte_calculations = 1
        if self.exposition == 'short':
            self.cte_calculations = -1
        self.spread_cost = spread_cost
        self.trading_cost = trading_cost
        database = Database().get_info(
            tickers=ticker, open_date=open_date, close_date=close_date)
        self.ticker_assets = database[[ticker+'_close']]
        self.daily_pnl_values=None
        self.pnl_value = None
        self.chages_checked = False

    def _premature_daily_pnl(self) -> pd.Series:
        first_value = self.ticker_assets.iloc[0][self.ticker+'_close']
        pnl_series = self.cte_calculations * \
            (self.ticker_assets-first_value)/first_value
        self._premature_pnl_values = pnl_series.rename(
            columns={self.ticker+'_close': 'daily_pnl'})
        return self._premature_pnl_values

    def _check_changes(self):
        daily_pnl = self._premature_daily_pnl()
        if self.chages_checked == True or (
                self.profit_margin == None and self.stop_loss == None):
            return
        dates = daily_pnl.index
        pnl_values = daily_pnl.values
        profit_margin = self.profit_margin
        if profit_margin == None:
            profit_margin = np.inf
        stop_loss = self.stop_loss
        if stop_loss == None:
            stop_loss = np.inf
        for i in range(len(daily_pnl)):
            pnl = pnl_values[i]
            if pnl >= profit_margin or pnl <= -stop_loss:
                self.close_date = dates[i]
                break
        self.ticker_assets = self.ticker_assets.loc[self.open_date:self.close_date]
        self.chages_checked = True

    def daily_pnl(self) -> pd.Series:
        if self.daily_pnl_values is not None:
            return self.daily_pnl_values
        self._check_changes()
        self.daily_pnl_values = self._premature_pnl_values.loc[
            self.open_date:self.close_date]
        distributed_cost = (self.spread_cost +
                            self.trading_cost)/len(self.daily_pnl_values)
        self.daily_pnl_values = self.daily_pnl_values - distributed_cost
        return self.daily_pnl_values

    def PnL(self) -> float:
        self._check_changes()
        if self.pnl_value is not None:
            return self.pnl_value
        self.pnl_value = (self.daily_pnl()['daily_pnl'][-1])
        return self.pnl_value

    def victory(self) -> bool:
        pnl = self.PnL()
        if pnl > 0:
            return True
        return False

    @property
    def last_date(self):
        self._check_changes()
        return self.close_date

    def daily_returns(self) -> pd.Series:
        self._check_changes()
        daily_returns = (self.ticker_assets -
                         self.ticker_assets.shift(1))/self.ticker_assets.shift(1)
        return daily_returns

    def __rpr__(self) -> str:
        return (self.ticker+' - '+self.exposition + ' - '+str(self.open_date)[:11]+' => '+str(self.close_date)[:11])
