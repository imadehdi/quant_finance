from class_option_pricer_MC import Option
import numpy as np


class BT_pricer(Option):
    def __init__(self, S0, r, sigma, T, K, call, american):
        super().__init__(S0, r, sigma, T, K, call, american)

    def pricing_option(self, N):
        S0 = self.S0
        r = self.r
        sigma = self.sigma
        T = self.T
        K = self.K
        call = self.call
        american = self.american

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        S = np.zeros((N + 1, N + 1))
        S[0, 0] = S0
        V = np.zeros_like(S)
        V[N, :] = np.maximum(S[N, :] - K, 0) if call else np.maximum(K - S[N, :], 0)
        for i in reversed(range(N)):
            for j in range(i + 1):
                V_up = V[i + 1, j + 1]
                V_down = V[i + 1, j]
                V_actualized = np.exp(-r * dt) * (p * V_up + (1 - p) * V_down)
                if american:
                    payoff = np.maximum(S[i, j] - K, 0) if call else np.maximum(K - S[i, j], 0)
                    V_actualized = max(payoff, V_actualized)
                V[i, j] = V_actualized
        C = V[0, 0]
        return C
