import pandas as pd
import numpy as np
import requests

def Get_Fred_Series(series_id='DTWEXM', api_key=None):
    if api_key is None:
        print('You need to register with the St Louis Fed and provide your API key to use this function')
        return ''
    
    import requests
 
    url = r'http://api.stlouisfed.org/fred/series/observations?series_id='+series_id+'&api_key='+api_key+'&file_type=json'
    response = requests.get(url)    
    response = response.json()
    
    dates = []
    values = []
    for obs in response['observations']:
        dates.append(pd.to_datetime(obs['date']))
        try:
            values.append(float(obs['value']))
        except:
            values.append(np.nan)
    return pd.Series(data=values,index=dates)