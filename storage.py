from .constants import *
from dotenv import load_dotenv
import os
import requests
import json
import datetime

class Storage:
    def __init__(self) -> None:
        load_dotenv()
        self.base_url = os.environ["FIREBASE_URL"]
        if self.base_url is None:
            raise Exception("Firebase link not found. Make sure you have a dedicated firebase project and pu the link in the .env file as 'FIREBASE_URL'.")
        today = (datetime.date.today())
        self.today_string = f'{str(today.year)}-{today.month}-{today.day}'
        
    def store_trades(self, params: list):
        info_dct = {}
        keys = TRADE_KEYS
        for info_index in range(len(params)):
            info_dct[keys[info_index]] = params[info_index]
        ticker = info_dct.pop('symbol')
        requests.post(
            f'{self.base_url}/trades/{self.today_string}/{ticker}.json', json.dumps(info_dct))
        
    def store_results(self,results:dict):
        requests.post(
            f'{self.base_url}/results/{self.today_string}.json', json.dumps(results))
        
    

