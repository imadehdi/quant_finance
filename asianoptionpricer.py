
import pandas as pd
import numpy as np

class AsianOption_MC_pricer:
    def __init__(self, options):
        self.options = options
    
    def MC_pricer(self, N, m):
        options = self.options
        n_options = len(options)
        S0 = options["S0"].values
        r = options["r"].values
        sigma = options["sigma"].values
        T = options["T"].values
        K = options["K"].values
        call = options["call"].values
        dt = T/m
        S = np.zeros((N, m + 1, n_options))
        S[:, 0, :] = S0
        for t in range(1, m+1):
            Z = np.random.randn(N, n_options)
            S[:, t, :] = S[:, t-1, :] * np.exp((r - 0.5 * (sigma**2)) * dt + sigma * np.sqrt(dt) * Z)
        
        S_mean = S.mean(axis=1)
        payoffs = np.where(call, np.maximum(S_mean - K, 0), np.maximum(K - S_mean, 0))
        C = np.exp(-r*T) * payoffs.mean(axis=0)
        self.options["price"] = pd.Series(C, index=self.options.index)
        