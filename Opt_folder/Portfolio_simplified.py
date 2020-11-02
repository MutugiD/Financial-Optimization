
import numpy as np
import pandas as pd
import cvxopt as opt
import cvxopt.solvers as optsolvers
import warnings


class Portfolio: 
    
    def markowitz_portfolio(cov_mat, exp_rets, target_ret,
                        allow_short=False, market_neutral=False):
                   
   
        if not isinstance(cov_mat, pd.DataFrame):
            raise ValueError("Covariance matrix is not a DataFrame")
    
        if not isinstance(exp_rets, pd.Series):
            raise ValueError("Expected returns is not a Series")
    
        if not isinstance(target_ret, float):
            raise ValueError("Target return is not a float")
    
        if not cov_mat.index.equals(exp_rets.index):
            raise ValueError("Indices do not match")
    
        if market_neutral and not allow_short:
            warnings.warn("A market neutral portfolio implies shorting")
            allow_short=True
    
        n = len(cov_mat)
    
        P = opt.matrix(cov_mat.values)
        q = opt.matrix(0.0, (n, 1))
    
        # Constraints Gx <= h
        if not allow_short:
            # exp_rets*x >= target_ret and x >= 0
            G = opt.matrix(np.vstack((-exp_rets.values,
                                      -np.identity(n))))
            h = opt.matrix(np.vstack((-target_ret,
                                      +np.zeros((n, 1)))))
        else:
            # exp_rets*x >= target_ret
            G = opt.matrix(-exp_rets.values).T
            h = opt.matrix(-target_ret)
    
        # Constraints Ax = b
        # sum(x) = 1
        A = opt.matrix(1.0, (1, n))
    
        if not market_neutral:
            b = opt.matrix(1.0)
        else:
            b = opt.matrix(0.0)
    
        # Solve
        optsolvers.options['show_progress'] = False
        sol = optsolvers.qp(P, q, G, h, A, b)
    
        if sol['status'] != 'optimal':
            warnings.warn("Convergence problem")
    
        # Put weights into a labeled series
        weights = pd.Series(sol['x'], index=cov_mat.index)
        return weights
    
    
    def min_var_portfolio(cov_mat, allow_short=False):
        
        if not isinstance(cov_mat, pd.DataFrame):
            raise ValueError("Covariance matrix is not a DataFrame")
    
        n = len(cov_mat)
    
        P = opt.matrix(cov_mat.values)
        q = opt.matrix(0.0, (n, 1))
    
        # Constraints Gx <= h
        if not allow_short:
            # x >= 0
            G = opt.matrix(-np.identity(n))
            h = opt.matrix(0.0, (n, 1))
        else:
            G = None
            h = None
    
        # Constraints Ax = b
        # sum(x) = 1
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
    
        # Solve
        optsolvers.options['show_progress'] = False
        sol = optsolvers.qp(P, q, G, h, A, b)
    
        if sol['status'] != 'optimal':
            warnings.warn("Convergence problem")
    
        # Put weights into a labeled series
        weights = pd.Series(sol['x'], index=cov_mat.index)
        return weights

    def tangency_portfolio(cov_mat, exp_rets, allow_short=False):
                
        if not isinstance(cov_mat, pd.DataFrame):
            raise ValueError("Covariance matrix is not a DataFrame")
    
        if not isinstance(exp_rets, pd.Series):
            raise ValueError("Expected returns is not a Series")
    
        if not cov_mat.index.equals(exp_rets.index):
            raise ValueError("Indices do not match")
    
        n = len(cov_mat)
    
        P = opt.matrix(cov_mat.values)
        q = opt.matrix(0.0, (n, 1))
    
        # Constraints Gx <= h
        if not allow_short:
            # exp_rets*x >= 1 and x >= 0
            G = opt.matrix(np.vstack((-exp_rets.values,
                                      -np.identity(n))))
            h = opt.matrix(np.vstack((-1.0,
                                      np.zeros((n, 1)))))
        else:
            # exp_rets*x >= 1
            G = opt.matrix(-exp_rets.values).T
            h = opt.matrix(-1.0)
    
        # Solve
        optsolvers.options['show_progress'] = False
        sol = optsolvers.qp(P, q, G, h)
    
        if sol['status'] != 'optimal':
            warnings.warn("Convergence problem")
    
        # Put weights into a labeled series
        weights = pd.Series(sol['x'], index=cov_mat.index)
    
        # Rescale weights, so that sum(weights) = 1
        weights /= weights.sum()
        return weights
    def max_ret_portfolio(exp_rets):
        
        if not isinstance(exp_rets, pd.Series):
            raise ValueError("Expected returns is not a Series")
    
        weights = exp_rets[:]
        weights[weights == weights.max()] = 1.0
        weights[weights != weights.max()] = 0.0
        weights /= weights.sum()
    
        return weights
    
    
    def truncate_weights(weights, min_weight=0.01, rescale=True):
        
        if not isinstance(weights, pd.Series):
            raise ValueError("Weight vector is not a Series")
    
        adj_weights = weights[:]
        adj_weights[adj_weights.abs() < min_weight] = 0.0
    
        if rescale:
            if not adj_weights.sum():
                raise ValueError("Cannot rescale weight vector as sum is not finite")
            
            adj_weights /= adj_weights.sum()
    
        return adj_weights
    def run(self):
        pass 
    
    
    
if __name__== '__main__':
    portfolio = Portfolio()
    portfolio.run()