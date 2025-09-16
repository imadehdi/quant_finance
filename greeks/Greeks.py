
import pandas as pd

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
    def __init__(self, options : pd.DataFrame):
        self.options = options.copy()
        


class Greeks:
    def __init__(self, options, pricer):
        self.options = options.copy()
        self.pricer = pricer
    
    def compute_delta(self, h : float=1e-3):
        options = self.options

        options_up = options.copy()
        options_up["S0"] += h

        options_down = options.copy()
        options_down["S0"] -= h 

        pricing_up = self.pricer(options_up)
        pricing_down = self.pricer(options_down)

        uk = [col for col in options if col != "S0"]
        pricing = pricing_up.merge(pricing_down, how="inner", on=uk, suffixes=["_up", "_down"])

        self.options["delta"] = (pricing["price_up"] - pricing["price_down"]) / (2*h)
        
    def compute_theta(self, dt : float=1e-3):
        options = self.options

        options_modif = options.copy()
        options_modif["T"] -= dt

        pricing_modif = self.pricer(options_modif)
        pricing_now = self.pricer(options)
        uk = [col for col in options if col != "T"]
        pricing = pricing_modif.merge(pricing_now, how="inner", on=uk, suffixes=["_modif", "_now"])

        self.options["theta"] = (pricing["price_modif"] - pricing["price_now"]) / (dt)

    def compute_gamma(self,  h : float=1e-3):
        options = self.options

        options_up = options.copy()
        options_up["S0"] += h

        options_down = options.copy()
        options_down["S0"] -= h 

        pricing_up = self.pricer(options_up)
        pricing_down = self.pricer(options_down)
        pricing_now = self.pricer(options)

        uk = [col for col in options if col != "S0"]
        pricing_v1 = pricing_up.merge(pricing_down, how="inner", on=uk, suffixes=["_up", "_down"])
        pricing = pricing_v1.merge(pricing_now, how="inner", on=uk, suffixes=["", "_now"])

        self.options["gamma"] = (pricing["price_up"] - 2 * pricing["price_now"] + pricing["price_down"]) / (h**2)



