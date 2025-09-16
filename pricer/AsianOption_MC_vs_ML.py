
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import matplotlib.pyplot as plt


################## Asian pricer based on MC ##################
class AsianOption_MC_pricer:
    def __init__(self, options):
        """
        Parameters : 
            - options : a dataframe which contains 6 columns (S0, r, sigma, T, K, call)
        """
        self.options = options
    
    def MC_pricer(self, N : int, m : int) -> pd.DataFrame:
        """
        Modifies the dataframe option by adding a column "price" which contains the price of the options based on the MC method. 
        TO DO : vectorization row 38.  
        Parameters : 
            - N : a int that represents the number of simulations
            - m : a int that represents the division of the time
        """
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
        return self.options


################## Data ##################
###### Generating ######
def generate_data(n, param_ranges, call_prob):
    """
    Generates data with np.random.uniform.
    Parameters : 
            - param_ranges : a dict that cotnains the range of each option's parameters
            - call_prob : a float that represents the probability of getting a call
    """
    S0 = np.random.uniform(param_ranges["S0"][0], param_ranges["S0"][1], n)
    K = np.random.uniform(param_ranges["K"][0], param_ranges["K"][1], n)
    T = np.random.uniform(param_ranges["T"][0], param_ranges["T"][1], n)
    r = np.random.uniform(param_ranges["r"][0], param_ranges["r"][1], n)
    sigma = np.random.uniform(param_ranges["sigma"][0], param_ranges["sigma"][1], n)
    call = np.random.rand(n) < call_prob

    df = pd.DataFrame({
        "S0": S0,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "call": call
    })

    return df

###### Formatting ######
def get_and_format_data(n, param_ranges, call_prob, N, m):
    """
    Gets and formats data for building the ML approximated pricer. 
    Parameters : 
            - param_ranges : a dict that cotnains the range of each option's parameters
            - call_prob : a float that represents the probability of getting a call
            - N : a int that represents the number of simulations in the MC pricer
            - m : a int that represents the division of the time in the MC pricer
    """
    options = generate_data(n, param_ranges, call_prob)
    pricer = AsianOption_MC_pricer(options)
    pricer.MC_pricer(N, m)
    options["call"] = options["call"].astype(int)
    return options

################## ML approximated pricer ##################
def create_approximated_pricer(options, n_estimators : int, max_depth : int, model_name : str, learning_rate : float):
    """
    Builds the approximated pricer based on a supervised algorithm. 
    Parameters : 
            - options : a dataframe which contains 6 columns (S0, r, sigma, T, K, call)
            - n_estimators : a int that represents the number of trees to be added. More estimators often mean more precision but also increase the risk of overfitting
            - max_depth : a int that represents the maximum depth of each tree. 
            - model_name : a str that represents the name of the model (RandomForest or HistGradient for now)
            - learning_rate : a float that represents the contribution of each tree in the final prediction. A smaller value means the model'll be more robust but also means requires more trees (used for histgradient)
    """
    X = options.drop(columns=["price"])
    y = options["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_name == "RandomForestRegressor":
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    if model_name == "HistGradientBoostingRegressor":
        model = HistGradientBoostingRegressor(max_iter=n_estimators, max_depth=max_depth,learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = mean_squared_error(y_test, y_pred, squared=False)
    
    return model, MAE, RMSE, y_test, y_pred


################## Example ##################
###### Parameters ######
param_ranges = {
    "S0": (50, 150),
    "K": (50, 150),
    "T": (0.1, 2.0),
    "r": (0.0, 0.05),
    "sigma": (0.1, 0.5)
}
n_options = 2000   
call_prob = 0.5    
N_mc = 5000      
m_mc = 50         

###### Hyperparamaters ######
n_estimators = 100
max_depth = 10
model_name = "RandomForestRegressor" 
learning_rate = 0.1

###### Computing ######
options = get_and_format_data(n_options, param_ranges, call_prob, N_mc, m_mc)
model, MAE, RMSE, y_test, y_pred = create_approximated_pricer(options, n_estimators, max_depth, model_name, learning_rate)

###### Plotting ######
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="y_test = y_pred")
plt.xlabel("Monte Carlo Price (y_test)")
plt.ylabel("Predicted Price (y_pred)")
plt.title("MC Price vs Predicted Price")
plt.legend()
plt.show()
