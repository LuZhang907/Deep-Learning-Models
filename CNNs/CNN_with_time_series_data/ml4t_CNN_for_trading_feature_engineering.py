import warnings
warnings.filterwarnings('ignore')

from talib import (RSI, BBANDS, MACD,
                   NATR, WILLR, WMA,
                   EMA, SMA, CCI, CMO,
                   MACD, PPO, ROC,
                   ADOSC, ADX, MOM)
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from pathlib import Path

DATA_STORE = '/lstr/sahara/mdep/luzhangstat/ml4t/data/assets.h5'

MONTH = 21
YEAR = 12 * MONTH

START = '2000-01-01'
END = '2017-12-31'

sns.set_style('whitegrid')
idx = pd.IndexSlice

T = [1, 5, 10, 21, 42, 63]

results_path = Path('/lstr/sahara/mdep/luzhangstat/ml4t/cnn_time_series', 'cnn_for_trading')
if not results_path.exists():
    results_path.mkdir(parents=True)

# Loading Quandl Wiki Stock Prices & Meta Data
adj_ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']

with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[START:END, :], adj_ohlcv]
              .rename(columns=lambda x: x.replace('adj_', ''))
              .swaplevel()
              .sort_index()
             .dropna())
    metadata = (store['us_equities/stocks'].loc[:, ['marketcap', 'sector']])
ohlcv = prices.columns.tolist()

prices.volume /= 1e3
prices.index.names = ['symbol', 'date']
metadata.index.name = 'symbol'

#Rolling universe: pick 500 most-traded stocks
dollar_vol = prices.close.mul(prices.volume).unstack('symbol').sort_index()
years = sorted(np.unique([d.year for d in prices.index.get_level_values('date').unique()]))

train_window = 5 # years
universe_size = 500

universe = []
for i, year in enumerate(years[5:], 5):
    start = str(years[i-5])
    end = str(years[i])
    most_traded = (dollar_vol.loc[start:end, :]
                   .dropna(thresh=1000, axis=1)
                   .median()
                   .nlargest(universe_size)
                   .index)
    universe.append(prices.loc[idx[most_traded, start:end], :])
universe = pd.concat(universe)

universe = universe.loc[~universe.index.duplicated()]
universe.info()

universe.groupby('symbol').size().describe()

universe.to_hdf('/lstr/sahara/mdep/luzhangstat/ml4t/data/assets.h5', 'universe')

#Generate Technical Indicators Factors
T = list(range(6, 21))


#Williams %R
for t in T:
    universe[f'{t:02}_WILLR'] = (universe.groupby(level='symbol', group_keys=False)
     .apply(lambda x: WILLR(x.high, x.low, x.close, timeperiod=t)))
    

#Normalized Average True Range
for t in T:
    universe[f'{t:02}_NATR'] = universe.groupby(level='symbol', 
                                group_keys=False).apply(lambda x: 
                                                        NATR(x.high, x.low, x.close, timeperiod=t))

#Commodity Channel Index
for t in T:    
    universe[f'{t:02}_CCI'] = (universe.groupby(level='symbol', group_keys=False)
     .apply(lambda x: CCI(x.high, x.low, x.close, timeperiod=t)))



#Chaikin A/D Oscillator
for t in T:
    universe[f'{t:02}_ADOSC'] = (universe.groupby(level='symbol', group_keys=False)
     .apply(lambda x: ADOSC(x.high, x.low, x.close, x.volume, fastperiod=t-3, slowperiod=4+t)))

#Average Directional Movement Index
for t in T:
    universe[f'{t:02}_ADX'] = universe.groupby(level='symbol', 
                                group_keys=False).apply(lambda x: 
                                                        ADX(x.high, x.low, x.close, timeperiod=t))

universe.drop(ohlcv, axis=1).to_hdf('data.h5', 'features')

##Compute Historical Returns

#Historical Returns
by_sym = universe.groupby(level='symbol').close
for t in [1,5]:
    universe[f'r{t:02}'] = by_sym.pct_change(t)

# Remove Outlier
universe[[f'r{t:02}' for t in [1, 5]]].describe()

outliers = universe[universe.r01>1].index.get_level_values('symbol').unique()
len(outliers)

universe = universe.drop(outliers, level='symbol')

#Rolling Factor Betas
factor_data = (web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', 
                              start=START)[0].rename(columns={'Mkt-RF': 'Market'}))
factor_data.index.names = ['date']

factor_data.info()

windows = list(range(15, 90, 5))
len(windows)

t = 1
ret = f'r{t:02}'
factors = ['Market', 'SMB', 'HML', 'RMW', 'CMA']
windows = list(range(15, 90, 5))
for window in windows:
    print(window)
    betas = []
    for symbol, data in universe.groupby(level='symbol'):
        model_data = data[[ret]].merge(factor_data, on='date').dropna()
        model_data[ret] -= model_data.RF

        rolling_ols = RollingOLS(endog=model_data[ret], 
                                 exog=sm.add_constant(model_data[factors]), window=window)
        factor_model = rolling_ols.fit(params_only=True).params.drop('const', axis=1)
        result = factor_model.assign(symbol=symbol).set_index('symbol', append=True)
        betas.append(result)
    betas = pd.concat(betas).rename(columns=lambda x: f'{window:02}_{x}')
    universe = universe.join(betas)

# Store Model Data
universe = universe.drop(ohlcv, axis=1)
universe.info()
drop_cols = ['r01',  'r05'] 


outcomes = universe.filter(like='_fwd').columns
universe = universe.sort_index()
with pd.HDFStore('/lstr/sahara/mdep/luzhangstat/ml4t/data/data.h5') as store:
    store.put('features', universe.drop(drop_cols, axis=1).drop(outcomes, axis=1).loc[idx[:, '2001':], :])
    store.put('targets', universe.loc[idx[:, '2001':], outcomes])