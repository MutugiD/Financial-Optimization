import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.cla import CLA
from matplotlib.ticker import FuncFormatter
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats
import math
import sys
import os

start_date = '2010-06-29'
end_date = '2020-08-10'

n = 5 
total_portfolio_value= 1000   
mu_min = 7  #minimum annual rate of return in percentages i.e 7% = 7
Q_min = 0.6  # minimum EGS score required by the investor i.e 60% =0.6

#change the DIRs
df_AAL = pd.read_csv(r'E:\Python Projects\Portfolio_Management\AAL.csv', index_col=0, parse_dates=True)
df_AAPL = pd.read_csv(r'E:\Python Projects\Portfolio_Management\AAPL.csv', index_col=0, parse_dates=True)
df_AMZN = pd.read_csv(r'E:\Python Projects\Portfolio_Management\AMZN.csv', index_col=0, parse_dates=True)
df_NFLX = pd.read_csv(r'E:\Python Projects\Portfolio_Management\NFLX.csv', index_col=0, parse_dates=True)
df_TSLA = pd.read_csv(r'E:\Python Projects\Portfolio_Management\TSLA.csv',index_col=0, parse_dates=True )
df_ESG= pd.read_csv(r'E:\Python Projects\Portfolio_Management\esg_sample.csv')
df_ESG['avg_ESG']= (df_ESG['environmental_score'] + df_ESG['social_score'] + df_ESG['gover_score'])/3




mean_avg_ESG = df_ESG['avg_ESG'].mean()
std_avg_ESG= df_ESG['avg_ESG'].std()
df_ESG['norm_ESG']=stats.norm.cdf((df_ESG['avg_ESG']-mean_avg_ESG)/std_avg_ESG)
Q = df_ESG['norm_ESG']

#if Q == 'NaN':  quit()   print ('NaN not allowed')


    


Adj_Close = df_AAL['Adj Close'], df_AAPL['Adj Close'],  df_AMZN['Adj Close'], df_NFLX['Adj Close'], df_TSLA['Adj Close']
df_Close = pd.DataFrame(Adj_Close, index = ['AAL', 'AAPL','AMZN', 'NFLX','TSLA']).T.loc[start_date:end_date]

log_returns = np.log(df_Close/df_Close.shift(1))
returns = df_Close.pct_change(1).iloc[1:]
returns_cov = np.array(returns.cov())
r = np.array(np.mean(returns, axis=0))
Sigma = risk_models.sample_cov(df_Close)
e = np.ones(len(r))

mu = 1+(mu_min/252)


def objective(w):
    return np.matmul(np.matmul(w,returns_cov),w)
w = np.random.random(len(r))
const = ({'type' : 'ineq' , 'fun' : lambda w: np.dot(w,r) + mu},
         {'type' : 'ineq' , 'fun' : lambda w: np.dot(w,Q) + Q_min}, 
         {'type' : 'eq' , 'fun' : lambda w: np.dot(w, e) - 1}) 

non_neg = []
for i in range(len(r)):
    non_neg.append((0,None))
non_neg = tuple(non_neg)

solution = minimize(fun=objective, x0=w, method='SLSQP',constraints=const,
                    bounds=non_neg)
w = solution.x.round(4)
print (w)
print (w.sum())
list = returns.columns[w > 0.0]
print(list)

#Rp = log_returns.mean()*252*w.sum() 
#std_dev = np.dot(np.dot(log_returns.cov()*252, w))

