from ..database import Database
import numpy as np

def ibov_related(
        ticker:str, 
        start_date: str = None, 
        end_date: str = None, 
        inferior_limit: float = 0.3, 
        superior_limit: float = 0.7
    ):
    data = Database()
    df = data.get_info(['RET_'+ticker, 'RET_IBOV'],start_date,end_date)
    df = df.dropna()
    columns = df.columns
    tck_values = (df[columns[0]].values > 0).astype(int)
    ibov_values = (df[columns[1]].values > 0).astype(int)
    is_equal = np.sum(tck_values == ibov_values)/len(tck_values)
    return is_equal > superior_limit or is_equal < inferior_limit
