'''
This Program imports real market data for the specified stocks and estimates the drift and volatility for our
geometric Brownian motion (GBM) model of the stock prices.
'''

import pandas as pd
import numpy as np

import matplotlib as plt
#import pandas_datareader as pdr
import yfinance as yf
#import datetime as dt

# Getting stock data from Yahoo finance and getting log-returns.

tickers = ['AAPL', 'IBM']
start_date = '2010-01-01'
end_date = '2020-01-01'
#data = pdr.get_data_yahoo(tickers, start_date, end_date)
data = yf.download(tickers, start_date, end_date, interval='1d')
data = data['Adj Close']
print(data.head())
print(data.tail())
#print(data.shift(freq = 'd').head())
log_returns = np.log(data/data.shift())
log_returns.dropna(inplace = True)
#print(log_returns.tail())

print(log_returns.describe())
#print(log_returns.mean()[0])
#print(log_returns.std())
b = [(log_returns.mean()[i] + 0.5*(log_returns.std()[i])**2) for i in tickers]
sig = log_returns.corr().values          # Currently correlation matrix, later transformed to sigma
for i in range(len(tickers)):
    for j in range(len(tickers)):
        sig[i,j] = sig[i,j]*log_returns.std()[i]*log_returns.std()[j]
#print(sigma)
'''So far, we have b and \sigma\sigma^T. sigma = sqrt(\sigma\sigma^T) through SVD.'''
sig = np.ndarray.tolist(np.linalg.cholesky(sig))
#print(sig)
out = {'drift':b, 'volatility':sig}

file = open('Data_files/b_and_sig.dat', 'w')
print(repr(out))
file.writelines([repr(out), '\n', repr(np.ndarray.tolist(data.iloc[-1,:].values))])
#print(out)

#verify that data.std() works properly
#print(np.matmul(sigma,sigma.T))

#print(log_returns.rolling(14).mean())
#print(data.values[1][0]/data.values[0][0])