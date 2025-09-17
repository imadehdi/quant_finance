import numpy as np
import pandas as pd


class BT_pricer:
    def __init__(self, options : pd.DataFrame):
        """
        Parameters : 
            - options : a dataframe which contains 6 columns (S0, r, sigma, T, K, call)
        """
        self.options = options
        required_cols = ["S0", "r", "sigma", "T", "K", "call", "american"]
        missing_cols = list(set(required_cols) - set(options.columns))
        if missing_cols:
            raise ValueError(f"Columns are missing : {missing_cols}")


    def pricing_option(self, N : int) -> pd.DataFrame:
        """
        Prices the options based on the Binomial Tree Method
        Parameters : 
            - N : a int that represents the deepth of the tree
        """
        options_copy = self.options.copy()
        S0 = options_copy["S0"].values
        r = options_copy["r"].values
        sigma = options_copy["sigma"].values
        T = options_copy["T"].values
        K = options_copy["K"].values
        call = options_copy["call"].values
        american = options_copy["american"].values
        n_options = len(options_copy)

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        S = np.zeros((n_options, N + 1, N + 1))
        S[:, 0, 0] = S0
        
        i = np.arange(N+1).reshape(-1, 1)
        j = np.arange(N+1).reshape(-1, 1)
        mask = j <= i

        for k in range(n_options):
            S_k = np.zeros((N+1, N+1))
            S_k[mask] = S0[k] * (u[k]**(i[mask]-j[mask])) * (d[k]**j[mask])
            S[k, :, :] = S_k


        V = np.zeros_like(S)
        V[:, N, :] = np.where(call, np.maximum(S[:, N, :] - K, 0), np.maximum(K - S[:, N, :], 0))
        for i in reversed(range(N)):
            for j in range(i + 1):
                V_up = V[:, i + 1, j + 1]
                V_down = V[:, i + 1, j]
                V_actualized = np.exp(-r * dt) * (p * V_up + (1 - p) * V_down)

                payoff = np.where(call, np.maximum(S[:, i, j] - K, 0), np.maximum(K - S[:, i, j], 0))
                V[:, i, j] = np.where(american, np.maximum(payoff, V_actualized), V_actualized)

        C = V[:, 0, 0]
        options_copy["price"] = pd.Series(C, index=self.options.index)
        return options_copy