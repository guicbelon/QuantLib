from quantitiative_finances import Database
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import warnings
warnings.simplefilter("ignore", UserWarning)

class LSTM_Forecast:
    def __init__(
            self, ticker: str, 
            open_date: str = None, 
            close_date: str = None, 
            periods_to_predict: int = 30, 
            number_of_targets: int = 3,
            train_split:float=0.2
        ) -> None:
        self.ticker = ticker
        self.periods_to_predict = periods_to_predict
        self.number_of_targets = number_of_targets
        self.train_split=train_split
        self.df = Database().get_info(
            'CRET_'+ticker, open_date=open_date, close_date=close_date)
        self.df = self.df.rename(columns={'CRET_'+ticker+'_close': 'Close'})
        self.open_date = self.df.index[0]
        self.close_date = self.df.index[-1] #- timedelta(days=periods_to_predict)
        self.model=None
        self.predicted_df=None

    def df_to_windowed_df(self,dataframe, first_date, last_date, n=3):
        target_date = first_date  
        dates = []
        X, Y = [], []
        last_time = False
        while True:
            df_subset = dataframe.loc[:target_date].tail(n+1)        
            if len(df_subset) != n+1:
                raise Exception(f'Error: Window of size {n} is too large for date {target_date}')
            values = df_subset['Close'].to_numpy()
            x, y = values[:-1], values[-1]
            dates.append(target_date)
            X.append(x)
            Y.append(y)
            next_week = dataframe.loc[target_date:target_date+timedelta(days=7)]
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            next_date_str = next_datetime_str.split('T')[0]
            year_month_day = next_date_str.split('-')
            year, month, day = year_month_day
            next_date = datetime(day=int(day), month=int(month), year=int(year))    
            if last_time:
                break    
            target_date = next_date
            if target_date == last_date:
                last_time = True    
        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates  
        X = np.array(X)
        for i in range(0, n):
            X[:, i]
            ret_df[f'Target-{n-i}'] = X[:, i]  
        ret_df['Target'] = Y
        return ret_df
    
    def windowed_df_to_date_X_y(self,windowed_dataframe):
        df_as_np = windowed_dataframe.to_numpy()
        dates = df_as_np[:, 0]
        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
        Y = df_as_np[:, -1]
        return dates, X.astype(np.float32), Y.astype(np.float32)
    
    def create_sessions(self):
        first_date = self.df.index[self.number_of_targets+1]
        last_date = self.close_date
        dataframe = self.df
        windowed_dataframe = self.df_to_windowed_df(dataframe,first_date,last_date)
        dates, X, Y = self.windowed_df_to_date_X_y(windowed_dataframe)
        train = int(len(dates)*self.train_split)
        validation = int(len(dates))
        dates_train, x_train, y_train = dates[:train], X[:train], Y[:train]
        dates_val, x_val, y_val = dates[train:validation], X[train:validation], Y[train:validation]
        sessions={
            'x_train':x_train,
            'x_val':x_val,
            'dates_train':dates_train,            
            'dates_val':dates_val,
            'y_train':y_train,
            'y_val':y_val
        }
        return sessions
    
    def get_model(self):
        if self.model is not None:
            return self.model
        sessions = self.create_sessions()
        x_train=sessions['x_train']
        y_train=sessions['y_train']
        x_val=sessions['x_val']
        y_val=sessions['y_val']  
        self.model = Sequential([layers.Input((3, 1)),
                            layers.LSTM(64),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(1)])
        self.model.compile(loss='mse', 
                    optimizer=Adam(learning_rate=0.001),
                    metrics=['mean_absolute_error'])
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100)
        return self.model
    
    def predict(self,input):
        model = self.get_model()
        predictions = model.predict(input).flatten()
        return predictions
    
    def future_predictions(self):
        if self.predicted_df is not None:
            return self.predicted_df
        begin = self.df.index[-(self.number_of_targets+1)]
        end=self.close_date +timedelta(days=(3*self.periods_to_predict))
        dates = (pd.bdate_range(begin,end))[:self.periods_to_predict]
        dates=(dates.strftime('%Y-%m-%d'))
        dates = pd.to_datetime(dates, format="%Y/%m/%d")
        dct_to_predict = {
            'date': dates,
            'Close': list(self.df['Close'][-(self.number_of_targets+1):]
                        )+[None]*(len(dates)-(self.number_of_targets+1))
        }
        df_to_predict = pd.DataFrame(dct_to_predict)
        df_to_predict = df_to_predict.set_index('date')

        for date_index in range(self.number_of_targets, len(dates)-1):
            previous_date = dates[date_index-self.number_of_targets]
            date = dates[date_index]
            next = (df_to_predict[previous_date:date])
            widowed_df = (self.df_to_windowed_df(next, date, date))
            new_date, X, Y = self.windowed_df_to_date_X_y(widowed_df)
            predicted_value=self.predict(X)[0]
            new_date=dates[date_index+1]
            future_date=(dates[date_index+1].strftime('%Y-%m-%d'))
            df_to_predict.xs(new_date)['Close'] = predicted_value
        self.predicted_df = df_to_predict
        return df_to_predict
