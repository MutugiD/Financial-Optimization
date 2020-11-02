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

start_date = '2005-09-27'
end_date = '2020-08-10'

target_return = 0.01 #annualized return
Q_e_min = 5
Q_s_min = 6
Q_g_min= 5.5

df_AAL = pd.read_csv(r'E:\Python Projects\Portfolio_Management_Jason\Data\dataset\AAL.csv', index_col=0, parse_dates=True)
df_AAPL = pd.read_csv(r'E:\Python Projects\Portfolio_Management_Jason\Data\dataset\AAPL.csv', index_col=0, parse_dates=True)
df_AMZN = pd.read_csv(r'E:\Python Projects\Portfolio_Management_Jason\Data\dataset\AMZN.csv', index_col=0, parse_dates=True)
df_NFLX = pd.read_csv(r'E:\Python Projects\Portfolio_Management_Jason\Data\dataset\NFLX.csv', index_col=0, parse_dates=True)
df_MSFT = pd.read_csv(r'E:\Python Projects\Portfolio_Management_Jason\Data\dataset\MSFT.csv',index_col=0, parse_dates=True )
df_ESG= pd.read_csv(r'E:\Python Projects\Portfolio_Management_Jason\esg_same_scores.csv')
#df_ESG['avg_ESG']= (df_ESG['environmental_score'] + df_ESG['social_score'] + df_ESG['gover_score'])/3
#mean_avg_ESG = df_ESG['avg_ESG'].mean()
#std_avg_ESG= np.std(df_ESG['avg_ESG'])
#df_ESG['norm_ESG']=stats.norm.cdf((df_ESG['avg_ESG']-mean_avg_ESG)/std_avg_ESG)
#Q = df_ESG['norm_ESG']

Adj_Close = df_AAL['Adj Close'], df_AAPL['Adj Close'],  df_AMZN['Adj Close'], df_MSFT['Adj Close'], df_NFLX['Adj Close']
df_Close = pd.DataFrame(Adj_Close, index = ['AAL', 'AAPL','AMZN', 'MSFT','NFLX']).T.loc[start_date:end_date]


log_returns = np.log(df_Close/df_Close.shift(1))
returns = df_Close.pct_change(1).iloc[1:]
returns_cov = np.array(returns.cov())
r = np.array(np.mean(returns, axis=0))
Sigma = risk_models.sample_cov(df_Close)
e = np.ones(len(r))

df_ESG['Q_e'] = stats.norm.cdf((df_ESG['environmental_score'] - df_ESG['environmental_score'].mean())/np.std(df_ESG['environmental_score']))
df_ESG['Q_s'] = stats.norm.cdf((df_ESG['social_score'] - df_ESG['social_score'].mean())/np.std(df_ESG['social_score']))
df_ESG['Q_g'] = stats.norm.cdf((df_ESG['gover_score'] - df_ESG['gover_score'].mean())/np.std(df_ESG['gover_score']))

Q_e, Q_s, Q_g = df_ESG['Q_e'], df_ESG['Q_s'], df_ESG['Q_g'] 

def objective(w):
    a = np.dot(w,r)
    b = np.log(1+a)  
    c = np.mean(b)
    return np.negative(c)
w = np.zeros(len(r)) + 1 / len(r)

if df_ESG.iloc[:, [3,4,5]].isnull().values.any() == False:
    const = ({'type' : 'ineq' , 'fun' : lambda w: np.dot(w,r) - target_return},         
             {'type' : 'eq' , 'fun' : lambda w: np.dot(w, e) - 1})
    
elif df_ESG.iloc[:, [3,4,5]].isnull().values.any() == False:
    const = ({'type' : 'ineq' , 'fun' : lambda w: np.dot(w,Q_e) - Q_e_min},
             {'type' : 'ineq' , 'fun' : lambda w: np.dot(w,Q_s) - Q_s_min},
             {'type' : 'ineq' , 'fun' : lambda w: np.dot(w,Q_g) - Q_g_min},
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
print(returns.columns[w > 0.0])
print(Q_e, Q_s, Q_g)



