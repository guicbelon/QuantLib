import MetaTrader5 as mtrader
from .storage import Storage
import time
import numpy as np
import warnings
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', UserWarning)

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Trader(metaclass=Singleton):
    def __init__(self, 
                 spread_percentage_limit=0.004, 
                 exposure=0.05, 
                 only_fractions=True
            ) -> None:
        mtrader.initialize()
        self.storage = Storage()
        symbols_list = []
        symbols = mtrader.symbols_get()
        for symbol in symbols:
            symbols_list.append(symbol.basis)
        self.symbols_list = list(set(symbols_list))
        self.all_tickers = self.symbols_list
        filtered_symbols = []
        if spread_percentage_limit is not None:
            for ticker in self.symbols_list:
                ticker_to_check = ticker
                if only_fractions:
                    ticker_to_check +='F'
                ticker_info = mtrader.symbol_info(ticker_to_check)
                if ticker_info is not None:
                    buy_price = round(ticker_info.ask,2)
                    sell_price = round(ticker_info.bid,2)
                    last_price = round(ticker_info.last,2)
                    if last_price!=0:
                        spread = buy_price-sell_price
                        spread_percentage = spread/last_price
                        spread_condition =  np.absolute(spread_percentage) <= spread_percentage_limit
                        interval_condition = buy_price>=last_price and sell_price<=last_price
                        if spread_condition and interval_condition:
                            filtered_symbols.append(ticker)
            self.symbols_list = filtered_symbols
        if only_fractions:
            for ticker in self.symbols_list:
                mtrader.symbol_select(ticker+'F', True)
        else:
            for ticker in self.symbols_list:
                mtrader.symbol_select(ticker, True)
        self.exposure = exposure
        self.only_fractions = only_fractions

    @property
    def account_info(self):
        return mtrader.account_info()
    
    @property
    def positions(self):
        return mtrader.positions_get()
    
    @property
    def cash(self):
        return mtrader.account_info().balance

    def send_order(self,
                   ticker: str, operation: str = 'buy' or 'sell',
                   volume: float = None, stop_loss: float = None,
                   profit_margin: float = None, comment: str = ''
                   ):
        info = mtrader.symbol_info(ticker)
        if info is None:
            raise Exception(f'{ticker} not found!')
        specific_info = mtrader.symbols_get(ticker)[0]
        if volume is None:
            volume = specific_info.volumelow
        order_dict = {'buy': 0, 'sell': 1}
        price_dict = {'buy': info.ask, 'sell': info.bid}
        request = {
            "action": mtrader.TRADE_ACTION_DEAL,
            "symbol": ticker,
            "volume": float(volume),
            "type": order_dict[operation],
            "price": price_dict[operation],
            "deviation": 20,
            "magic": round(time.time()*1000),
            "comment": comment,
            "type_time": mtrader.ORDER_TIME_GTC,
            "type_filling": mtrader.ORDER_FILLING_RETURN,
        }
        if stop_loss is not None:
            request["sl"] = price_dict[operation](1-stop_loss)
        if profit_margin is not None:
            request["tp"] = price_dict[operation](1+profit_margin)
        order = mtrader.order_send(request)
        if order is None:
            raise Exception(
                'Error in sending request. MetaTrader5.order_send returned None.')
        if order.retcode != mtrader.TRADE_RETCODE_DONE:
            raise Exception(
                f'Error in sending request. Error code: {order.retcode}.')
        order_info = list(order)[:-1]+list(order[-1])
        self.storage.store_trades(order_info)

    def alocate(self,
                ticker: str, num_of_tickers: int, operation: str = 'buy' or 'sell',
                volume: float = None, stop_loss: float = None,
                profit_margin: float = None, comment: str = ''
                ):
        if self.only_fractions:
            ticker += 'F'
        try:
            available_cash = self.cash*self.exposure
            maximum_to_spend = available_cash/num_of_tickers
            symbol_get = mtrader.symbols_get(ticker)
            ticker_price = symbol_get[0].last
            volume = maximum_to_spend//ticker_price
            self.send_order(ticker, operation, volume,
                            stop_loss, profit_margin, comment)
        except:
            pass

    def get_tickers(self, ticker: str = None):
        if ticker is None:
            return self.symbols_list
        else:
            symbols = mtrader.symbols_get(ticker)
        symbols_list = []
        for symbol in symbols:
            symbols_list.append(symbol.basis)
        symbols_list = list(set(symbols_list))
        return symbols_list

    def close_ticker_order(self, ticker):
        positions = mtrader.positions_get()
        for position in positions:
            symbol = position.symbol
            if ticker == symbol:
                ticket = position.ticket
                mtrader.Close(symbol=symbol, ticket=ticket)
                break

    def close_all_orders(self, record = True):
        positions = mtrader.positions_get()
        if len(positions)==0:
            return
        total_profit = 0
        hit = 0
        specific_info = {}
        for position in positions:
            symbol = position.symbol
            profit = position.profit
            total_profit += profit
            action = 'buy' if position.type == 0 else 'sell'
            if profit > 0:
                hit += 1
            specific_info[symbol] = {
                'profit': profit,
                'action': action,
                'volume': position.volume,
                'open_price': position.price_open,
                'close_price': position.price_current,
                'comment': position.comment,
                'stop_loss': position.sl,
                'take_profit': position.tp
            }            
            ticket = position.ticket
            mtrader.Close(symbol=symbol, ticket=ticket)
        results={
            'total_profit': total_profit,
            'hit_ratio': hit/len(positions),
            'specific_info':specific_info,
            'assets':self.cash
        }
        if record:
            self.storage.store_results(results)
