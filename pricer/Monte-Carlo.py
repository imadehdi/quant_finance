
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class Options:
    def __init__(self, options : pd.DataFrame):
        """
        Parameters : 
            - options : a dataframe which contains 7 columns (S0, r, sigma, T, K, call, american)
        """
        self.options = options
    
    def divide_EUR_US_options(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divises the df options into two dfs (EUR vs US).
        """
        EUR_options = self.options[~self.options["american"]]
        US_options = self.options[self.options["american"]]
        return EUR_options, US_options


class MC_priceur(Options):
    def __init__(self, options : pd.DataFrame):
        super().__init__(options)

    def pricing_european_options_MC(self, 
                                    EUR_options : pd.DataFrame, 
                                    N : int, 
                                    S0_v : pd.Series, 
                                    r_v : pd.Series, 
                                    sigma_v : pd.Series, 
                                    T_v : pd.Series, 
                                    K_v : pd.Series, 
                                    call_v : pd.Series, 
                                    american_v : pd.Series) -> pd.DataFrame:
        """
        Prices the european options based on a MC method.
        Parameters : 
            - EUR_options : a dataframe which only contains european options
            - N : an int which represents the number of simulations (uniq)  
            - S0_v, ..., K_v : vectors which contains their value of each option
            - call_v : a vector which contains True if the option is a call, otherwise False 
            - american_v : a vector which contains True if the option is a american one, otherwie False 
        """
        options_copy = EUR_options.copy()
        
        if american_v.any():
            raise ValueError("This vectorized method only works on european options")
        n_options = len(options_copy)

        Z = np.random.randn(N, n_options)
        ST = S0_v * np.exp((r_v - 0.5 * (sigma_v**2)) * T_v + sigma_v * np.sqrt(T_v) * Z)
        payoffs = np.where(call_v, np.maximum(ST - K_v, 0), np.maximum(K_v - ST, 0))
        C = np.exp(-r_v * T_v) * payoffs.mean(axis=0)
        options_copy["price"] = pd.Series(C, index=EUR_options.index)
        return options_copy
    
    def pricing_american_options_MC(self, 
                                    US_options : pd.DataFrame, 
                                    N : int, 
                                    m : int,
                                    S0_v : pd.Series, 
                                    r_v : pd.Series, 
                                    sigma_v : pd.Series, 
                                    T_v : pd.Series, 
                                    K_v : pd.Series, 
                                    call_v : pd.Series, 
                                    american_v : pd.Series) -> pd.DataFrame:
        """
        Prices the american options based on a MC method.
        Parameters : 
            - US_options : a dataframe which only contains US options
            - N : an int which represents the number of simulations (uniq)
            - m : an int which represents the way we divide the time of the options (uniq)
            - S0_v, ..., K_v : vectors which contains their value of each option
            - call_v : a vector which contains True if the option is a call, otherwise False 
            - american_v : a vector which contains True if the option is a american one, otherwie False 
        """
        options_copy = US_options.copy()
        prices = pd.Series(index=US_options.index, dtype=float)
        if not american_v.any():
            raise ValueError("This method only works on american options")
        for i in US_options.index:
            S0 = S0_v.loc[i]
            r = r_v.loc[i]
            sigma = sigma_v.loc[i]
            T = T_v.loc[i]
            K = K_v.loc[i]
            call = call_v.loc[i]
            american = american_v.loc[i]
            if american:
                dt = T / m
                Z = np.random.randn(N, m)
                S = np.zeros((N, m + 1))
                payoffs = np.zeros((N, m))

                S[:, 0] = S0
                S[:, 1:] = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))

                payoffs = np.maximum(S[:, :-1] - K, 0) if call else np.maximum(K - S[:, :-1], 0)
                itm = payoffs > 0
            
                for k in reversed(range(m)):
                    if itm[:, k].any():
                        X = S[itm[:, k], k].reshape(-1, 1)
                        y = payoffs[itm[:, k], k + 1] * np.exp(-r * dt)

                        model = LinearRegression()
                        model.fit(X, y)
                        pred_val = model.predict(X)

                        payoffs_immediat = np.maximum(S[itm[:, k], k] - K, 0) if call else np.maximum(K - S[itm[:, k], k], 0)
                        payoffs[itm[:, k], k] = np.where(payoffs_immediat > pred_val, payoffs_immediat, pred_val)
                        payoffs[~itm[:, k], k] = payoffs[~itm[:, k], k + 1] * np.exp(-r * dt)
                prices[i] = payoffs[:, 0].mean()
        options_copy["price"] = prices
        return options_copy


    def pricing_european_american_option_MC(self, N, m):
        EUR_options, US_options = self.divide_EUR_US_options()
        options_copy = self.options.copy()
        
        prices = pd.Series(index=options_copy.index, dtype=float)

        if not EUR_options.empty:
            S0_v_EUR = EUR_options["S0"].values
            r_v_EUR = EUR_options["r"].values
            sigma_v_EUR = EUR_options["sigma"].values
            T_v_EUR = EUR_options["T"].values
            K_v_EUR = EUR_options["K"].values
            call_v_EUR = EUR_options["call"].values
            american_v_EUR = EUR_options["american"].values
            df_EUR_priced = self.pricing_european_options_MC(EUR_options, N, S0_v_EUR, r_v_EUR, sigma_v_EUR, T_v_EUR, K_v_EUR, call_v_EUR, american_v_EUR)
            prices.loc[df_EUR_priced.index] = df_EUR_priced["price"]
            
        if not US_options.empty:
            S0_v = options_copy["S0"]
            r_v = options_copy["r"]
            sigma_v = options_copy["sigma"] 
            T_v = options_copy["T"] 
            K_v = options_copy["K"]
            call_v = options_copy["call"] 
            american_v = options_copy["american"] 
            
            df_US_priced = self.pricing_american_options_MC(US_options, N, m, S0_v, r_v, sigma_v, T_v, K_v, call_v, american_v)
            prices.loc[df_US_priced.index] = df_US_priced["price"]
            
        options_copy["price"] = prices
        return options_copy