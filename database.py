from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import sgs
from bcb import sgs as sgs_bcb


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Database(metaclass=Singleton):
    def __init__(self) -> None:
        self._DATA = pd.DataFrame()
        self._seeken_dates = {}
        load_dotenv()
        self.fintz_key = os.environ["FINTZ_KEY"]
        if self.fintz_key is None:
            raise Exception(
                "Fintz API key not found! Make sure you have access to this API and set it's key in the .env file as 'FINTZ_KEY'.")

    def _add_seeken_dates(self, ticker, open_date, close_date):
        dct_dates = {
            'start': open_date,
            'close': close_date
        }
        self._seeken_dates[ticker] = dct_dates

    def _fetch_selic(self, open_date, close_date):
        open_date_string = (str(open_date.day)+'%2F' +
                            str(open_date.month)+'%2F'+(str(open_date.year)))
        close_date_string = (str(close_date.day)+'%2F' +
                             str(close_date.month)+'%2F'+(str(close_date.year)))
        url_request = ("""https://brapi.dev/api/v2/prime-rate?country=brazil&historical=true&start={}&end={}&sortBy=date&sortOrder=asc"""
                       ).format(open_date_string, close_date_string)
        rqst = requests.get(url_request)
        obj = json.loads(rqst.text)
        data = obj['prime-rate']
        dates = []
        selic = []
        for daily_data in data:
            dates.append(daily_data['date'])
            selic.append(float(daily_data['value']))
        dates = pd.to_datetime(dates, format="%d/%m/%Y")
        info_dct = {
            'date': dates,
            'SELIC_close': selic
        }
        df = pd.DataFrame(info_dct)
        df = df.set_index('date')
        df = df.loc[open_date:close_date]
        self._DATA = pd.concat([df, self._DATA], axis=1)

    def _fetch_CDI(self, open_date, close_date):
        open_date_string = (str(open_date.day)+'/' +
                            str(open_date.month)+'/'+(str(open_date.year)))
        close_date_string = (str(close_date.day)+'/' +
                             str(close_date.month)+'/'+(str(close_date.year)))
        data_cdi = sgs.time_serie(
            12, start=open_date_string, end=close_date_string)
        df = pd.DataFrame(data_cdi)
        df = df.rename(columns={12: 'CDI_close'})
        df = df.loc[open_date:close_date]
        self._DATA = pd.concat([df, self._DATA], axis=1)

    def _fetch_IPCA_IGPM(self, ticker, open_date, close_date):
        info = {'IPCA': 433, 'IGPM': 189}
        code = info[ticker]
        df = sgs_bcb.get(code, start=open_date, end=close_date)
        df = df.rename(columns={str(code): ticker+'_close'})
        df = df.loc[open_date:close_date]
        self._DATA = pd.concat([df, self._DATA], axis=1)

    def _fetch_yf(self, ticker: str, open_date, close_date, interval='1d'):
        ticker_yf = yf.Ticker(ticker+'.SA')
        if ticker =='IBOV':
            ticker_yf = yf.Ticker("^BVSP")
        if ticker =='SPX':
            ticker_yf = yf.Ticker("^GSPC")        
        close_to_seek = close_date + timedelta(days=10)
        candles = ticker_yf.history(
            start=open_date, end=close_to_seek, interval=interval)
        if len(candles) == 0:
            ticker_yf = yf.Ticker(ticker)
            candles = ticker_yf.history(
                start=open_date, end=close_to_seek, interval=interval)
            if len(candles) == 0:
                return False
        candles = candles.rename(
            columns={'Open': ticker+'_open', 'High': ticker + '_high',
                     'Low': ticker + '_low', 'Close': ticker + '_close',
                     'Volume': ticker + '_volume'})
        candles.index.names = ['date']
        candles = candles.tz_localize(None)
        candles = candles[[ticker+'_close', ticker+'_open',
                           ticker+'_high', ticker+'_low', ticker+'_volume']]
        self._DATA = pd.concat([candles, self._DATA], axis=1)
        return True

    def _fetch_brapi(self, ticker: str, open_date, close_date):
        url_request = (
            f"https://brapi.dev/api/quote/{ticker}?range=max&interval=1d&fundamental=false")
        rqst = requests.get(url_request)
        obj = json.loads(rqst.text)
        error = obj.get('error')
        if error:
            return False
        data = obj['results'][0]['historicalDataPrice']
        dates = []
        open = []
        close = []
        high = []
        low = []
        volume = []
        for daily_data in data:
            dates.append(daily_data['date'])
            open.append(daily_data['open'])
            close.append(daily_data['close'])
            high.append(daily_data['high'])
            low.append(daily_data['low'])
            volume.append(daily_data['volume'])
        dates = [datetime.fromtimestamp(ts) for ts in dates]
        dates = [dt.replace(hour=0, minute=0, second=0) for dt in dates]
        dates = pd.to_datetime(dates)
        info_dct = {
            'date': dates,
            ticker+'_open': open,
            ticker+'_high': high,
            ticker+'_low': low,
            ticker+'_close': close,
            ticker+'_volume': volume,
        }
        df = pd.DataFrame(info_dct)
        df = df.set_index('date')
        df = df.loc[open_date:close_date]
        self._DATA = pd.concat([df, self._DATA], axis=1)
        return True

    def _fetch_prices(self, ticker, open_date, close_date):
        data_to_fetch = {
            'ticker': ticker,
            'open_date': open_date,
            'close_date': close_date
        }
        yahoo = self._fetch_yf(**data_to_fetch)
        if not yahoo:
            brapi = self._fetch_brapi(**data_to_fetch)
            if not brapi:
                raise Exception("""No data found for {}!""".format(ticker))

    def _fetch_currencies(self, ticker, open_date, close_date):
        splited_ticker = ticker.split('/')
        ticker_to_fetch = splited_ticker[0]+splited_ticker[1]+'=X'
        days_to_seek = (pd.to_datetime(date.today()) - open_date).days +10
        data = yf.download(
            ticker_to_fetch, period=f"{str(days_to_seek)}d", progress=False)
        data = data.rename(
            columns={'Open': ticker+'_open', 'High': ticker + '_high',
                     'Low': ticker + '_low', 'Close': ticker + '_close',
                     'Volume': ticker + '_volume'})
        data.index.names = ['date']
        data = data.tz_localize(None)
        data = data[[ticker+'_close', ticker+'_open',
                           ticker+'_high', ticker+'_low', ticker+'_volume']]
        self._DATA = pd.concat([data, self._DATA], axis=1)

    def _fetch_returns(self, ticker, open_date, close_date):
        data = self._DATA[[ticker+'_close']]
        daily_retuns = (data-data.shift(1))/data.shift(1)
        df = pd.DataFrame(daily_retuns)
        df = df.rename(columns={ticker+'_close': 'RET_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        self._DATA = pd.concat([df, self._DATA], axis=1)

    def _fetch_log_returns(self, ticker, open_date, close_date):
        data = self._DATA[[ticker+'_close']]
        daily_retuns = np.log((data)/data.shift(1))
        df = pd.DataFrame(daily_retuns)
        df = df.rename(columns={ticker+'_close': 'LRET_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        self._DATA = pd.concat([df, self._DATA], axis=1)

    def _fetch_cumulated_returns(self, ticker, open_date, close_date):
        data = self._DATA[[ticker+'_close']]
        daily_retuns = (data-data.shift(1))/data.shift(1)
        cumulated_returns = (daily_retuns.fillna(0)+1).cumprod()-1
        df = pd.DataFrame(cumulated_returns)
        df = df.rename(columns={ticker+'_close': 'CRET_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        self._DATA = pd.concat([df, self._DATA], axis=1)

    def _fetch_cumulated_log_returns(self, ticker, open_date, close_date):
        data = self._DATA[[ticker+'_close']]
        daily_retuns = np.log((data)/data.shift(1))
        cumulated_returns = (daily_retuns.fillna(0)).cumsum()
        df = pd.DataFrame(cumulated_returns)
        df = df.rename(columns={ticker+'_close': 'CLRET_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        self._DATA = pd.concat([df, self._DATA], axis=1)

    def _fetch_volatility(self, ticker, periods, open_date, close_date):
        data = self._DATA[[ticker+'_close']].dropna()
        daily_retuns = (data-data.shift(1))/data.shift(1)
        volatility = (daily_retuns.rolling(periods).std())*(periods**(1/2))
        df = pd.DataFrame(volatility)
        df = df.rename(columns={ticker+'_close': 'VOL' +
                       str(periods)+'_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        self._DATA = pd.concat([df, self._DATA], axis=1)

    def _check_index(self, ticker):
        info_dct = {'ticker': ticker, 'transf': None,
                    'get_prices': True, 'previous_days': None, 'currencies': False}
        ticker_splitted = ticker.split('_')
        if ticker_splitted[0][:3] == 'VOL':
            info_dct['ticker'] = ticker_splitted[1]
            info_dct['transf'] = 'VOL'
            info_dct['periods'] = int(ticker_splitted[0][3:])
            info_dct['previous_days'] = (int(ticker_splitted[0][3:])+150)
        elif ticker_splitted[0] == 'RET' or ticker_splitted[0] == 'LRET':
            info_dct['ticker'] = ticker_splitted[1]
            info_dct['transf'] = ticker_splitted[0]
            info_dct['previous_days'] = None
        elif ticker_splitted[0] == 'CRET' or ticker_splitted[0] == 'CLRET':
            info_dct['ticker'] = ticker_splitted[1]
            info_dct['transf'] = ticker_splitted[0]
            info_dct['previous_days'] = None
        is_currency = len(ticker.split('/')) > 1
        if is_currency:
            info_dct['currencies'] = True
        is_selic = ticker_splitted[-1] == 'SELIC'
        is_cdi = ticker_splitted[-1] == 'CDI'
        is_ipca = ticker_splitted[-1] == 'IPCA'
        is_igpm = ticker_splitted[-1] == 'IGPM'
        is_ibov = ticker_splitted[-1] == 'IBOV'
        is_spx = ticker_splitted[-1] == 'SPX'
        if is_selic or is_cdi or is_ibov or is_spx or is_ipca or is_igpm or is_currency:
            info_dct['get_prices'] = False
        return info_dct

    def _allow_changes(self, ticker, open_date, close_date):
        ticker_data = self._check_index(ticker)
        if ticker_data['previous_days'] is not None:
            open_date = pd.to_datetime(
                open_date) - timedelta(days=(ticker_data['previous_days']))
        ticker = ticker_data['ticker']
        if ticker in self._seeken_dates.keys():
            previous_than_start = self._seeken_dates[ticker]['start'] > pd.to_datetime(
                open_date)
            after_close = self._seeken_dates[ticker]['close'] < pd.to_datetime(
                close_date)
            there_is_transf = ticker_data['transf'] is not None
            if previous_than_start or after_close or there_is_transf:
                self._DATA = self._DATA.drop(columns=[ticker+'_close'])
                try:
                    self._DATA = self._DATA.drop(columns=[ticker+'_open',
                                                          ticker+'_high', ticker+'_low', ticker+'_volume'])
                except:
                    pass
                try:
                    self._DATA = self._DATA.drop(
                        columns=[ticker_data['transf']+'_'+ticker+'_close'])
                except:
                    pass
                try:
                    self._DATA = self._DATA.drop(
                        columns=[ticker_data['transf']+str(ticker_data['periods'])+'_'+ticker+'_close'])
                except:
                    pass
                open_date = min(
                    self._seeken_dates[ticker]['start'], pd.to_datetime(open_date))
                close_date = max(
                    self._seeken_dates[ticker]['close'], pd.to_datetime(close_date))
                self._add_seeken_dates(ticker, open_date, close_date)
                return {'changes': True, 'open_date': open_date, 'close_date': close_date}
            else:
                return {'changes': False}
        self._add_seeken_dates(ticker, open_date, close_date)
        return {'changes': True, 'open_date': open_date, 'close_date': close_date}

    def _add_assets(self, ticker: str, open_date, close_date):
        changes_data = self._allow_changes(ticker, open_date, close_date)
        if not changes_data['changes']:
            return
        ticker_data = self._check_index(ticker)
        open_date = changes_data['open_date']
        close_date = changes_data['close_date']
        ticker = ticker_data['ticker']
        if ticker_data['get_prices']:
            self._fetch_prices(ticker, open_date, close_date)
        if ticker_data['currencies']:
            self._fetch_currencies(ticker, open_date, close_date)

        if ticker_data['ticker'] == 'SELIC':
            self._fetch_selic(open_date, close_date)
        elif ticker_data['ticker'] == 'CDI':
            self._fetch_CDI(open_date, close_date)
        elif ticker_data['ticker'] == 'IPCA' or ticker_data['ticker'] == 'IGPM':
            self._fetch_IPCA_IGPM(ticker_data['ticker'], open_date, close_date)
        elif ticker_data['ticker'] == 'IBOV' or ticker_data['ticker'] == 'SPX':
            self._fetch_yf(ticker_data['ticker'], open_date, close_date)

        if ticker_data['transf'] == 'VOL':
            self._fetch_volatility(
                ticker, ticker_data['periods'], open_date, close_date)
        elif ticker_data['transf'] == 'RET':
            self._fetch_returns(ticker, open_date, close_date)
        elif ticker_data['transf'] == 'LRET':
            self._fetch_log_returns(ticker, open_date, close_date)
        elif ticker_data['transf'] == 'CRET':
            self._fetch_cumulated_returns(ticker, open_date, close_date)
        elif ticker_data['transf'] == 'CLRET':
            self._fetch_cumulated_log_returns(ticker, open_date, close_date)

    def get_info(self, tickers, open_date: str = None, close_date: str = None, info='close' or 'ohlcv'):
        if close_date is None:
            close_date = pd.to_datetime(date.today())  # - timedelta(days=1)
        if open_date is None:
            open_date = '1950'
        open_date = pd.to_datetime(open_date)
        close_date = pd.to_datetime(close_date)
        if type(tickers) is str:
            tickers = [tickers]
        tickers_to_display = []
        for ticker in tickers:
            ticker = ticker.upper()
            self._add_assets(ticker, open_date, close_date)
            if info == 'ohlcv':
                tickers_to_display += [ticker+'_open', ticker+'_high',
                                       ticker+'_low', ticker+'_close',
                                       ticker+'_volume']
            else:
                tickers_to_display += [ticker+'_'+info]
        return self._DATA[tickers_to_display].loc[open_date:close_date]

    def reset(self):
        self._DATA = pd.DataFrame()
        self._seeken_dates = {}

    @property
    def data(self):
        return self._DATA

