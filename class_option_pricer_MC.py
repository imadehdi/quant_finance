import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class Option:
    def __init__(self, S0, r, sigma, T, K, call, american):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.call = call
        self.american = american


class Options_vectorized:
    def __init__(self, S0, r, sigma, T, K, call, american):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.call = call
        self.american = american


class MC_priceur(Option):
    def __init__(self, S0, r, sigma, T, K, call):
        super().__init__(S0, r, sigma, T, K, call, american=False)

    def pricing_european_option_MC(self, N):
        S0 = self.S0
        r = self.r
        sigma = self.sigma
        T = self.T
        K = self.K
        call = self.call

        Z = np.random.randn(N)
        ST = S0 * np.exp((r - 0.5 * (sigma**2)) * T + sigma * np.sqrt(T) * Z)
        payoffs = np.maximum(ST - K, 0) if call else np.maximum(K - ST, 0)
        C = np.exp(-r * T) * np.mean(payoffs)
        return C

    def pricing_american_option_MC(self, N, m):
        S0 = self.S0
        r = self.r
        sigma = self.sigma
        T = self.T
        K = self.K
        call = self.call

        dt = T / m
        Z = np.random.randn(N, m)
        S = np.zeros((N, m + 1))
        payoffs = np.zeros((N, m))

        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))

        payoffs = np.maximum(S[:, :-1] - K, 0) if call else np.maximum(K - S[:, :-1], 0)
        itm = payoffs > 0

        for k in reversed(range(m)):
            X = S[itm[:, k], k].reshape(-1, 1)
            y = payoffs[itm[:, k], k + 1] * np.exp(-r * dt)

            model = LinearRegression()
            model.fit(X, y)
            pred_val = model.predict(X)

            payoffs_immediat = np.maximum(S[itm[:, k], k] - K, 0) if call else np.maximum(K - S[itm[:, k], k], 0)
            payoffs[itm[:, k], k] = np.where(payoffs_immediat > pred_val, payoffs_immediat, pred_val)
            payoffs[~itm[:, k], k] = payoffs[~itm[:, k], k + 1] * np.exp(-r * dt)

        C = np.mean(payoffs[:, 0])
        return C
