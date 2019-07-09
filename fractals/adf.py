# Augmented Dickey-Fuller Tests
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from statsmodels.tsa.stattools import adfuller
pd.core.common.is_list_like = pd.api.types.is_list_like
import matplotlib.pyplot as plt
from datetime import datetime
import time
# %matplotlib inline

start, end = datetime(2016, 1, 1), time.strftime("%x")
aapl = pdr.DataReader(['AAPL'],
                      'yahoo',
                      start,
                      end)
aapl.columns = [col[0].lower().replace(' ', '_')
                for col in aapl.columns]

aapl_close = aapl[['close']]
aapl_close = aapl_close.apply(lambda x: np.log(x) - np.log(x.shift(1)))
aapl_close.dropna(inplace=True)

_ = aapl_close.plot(figsize=(20, 10),
                    linewidth=3,
                    fontsize=14)

result = adfuller(aapl_close['close'].values)
print('Augmented Dickey-Fuller test statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'. format(key, value))