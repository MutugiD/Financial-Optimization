import pandas as pd
import numpy as np
from datetime import datetime
import scipy as sp
import scipy.optimize as scopt
import scipy.stats as spstats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



class MPT:
    
    def create_portfolio(self, tickers, weights=None):
        if (weights is None):
            shares = np.ones(len(tickers))/len(tickers)
        #portfolio = pd.DataFrame({'Tickers': tickers, index=tickers) 'Weights': weights},
                        

                



             
        return portfolio
    def calculate_weighted_portfolio_value(self, portfolio,returns, name='Value'):
        total_weights = portfolio.Weights.sum()
        weighted_returns = returns * (portfolio.Weights / total_weights)
        return pd.DataFrame({name: weighted_returns.sum(axis=1)})
        
    def url_main(self, hmtl): 
            pass 
    def info_main(self): 
            pass 
    def csv(self, excel):
            pass
    def run(self): 
        portfolio = self.create_portfolio(['Stock A', 'Stock B'], [1, 1])
        returns = pd.DataFrame({'Stock A': [0.1, 0.24, 0.05, -0.02, 0.2],
                                'Stock B': [-0.15, -0.2, -0.01, 0.04, -0.15]})
        wr = self.calculate_weighted_portfolio_value(portfolio,returns,"Value")
        with_value = pd.concat([returns, wr], axis=1)
        with_value_std = with_value.std()
        print(with_value)
        
if __name__== '__main__':
    scrapper = MPT()
    scrapper.run()
        
        
        
        
        




        
        
        
    




            
    
#if __name__== '__main__':
#runMPT = MPT() runMPT.run()

    
