from scipy.sparse.linalg import svds
from collections import defaultdict
from random import random
import numpy as np


class LFM:
    def __init__(self, n_latent_factors=20, reg_q=10, reg_p=1, reg_bi=0.001, reg_bu=0.1,
                 eta_q=.006, eta_p=.006, eta_bi=.006, eta_bu=.006,
                 users=None, items=None, mode='Latent'):
        self.reg_q: float = reg_q
        self.reg_p: float = reg_p
        self.reg_bi: float = reg_bi
        self.reg_bu: float = reg_bu
        self.eta_q: float = eta_q
        self.eta_p: float = eta_p
        self.eta_bi: float = eta_bi
        self.eta_bu: float = eta_bu
        self.n_latent_factors: int = n_latent_factors
        self.train_dict: dict = {}
        self.Q: np.array = None
        self.P: np.array = None
        self.Bu: np.array = None
        self.Bi: np.array = None
        self.mean_rating: float = 0.0
        self.users = users
        self.items = items
        self.mode = mode

        if self.mode == 'Latent':
            self.utility_matrix = self._build_utility_matrix()
        else:
            # just use user/item biases
            self.utility_matrix = None

    def fit(self, train_dict, global_mean, user_averages, item_averages, num_epochs=10):
        np.random.seed(1)
        if self.mode == 'Latent':
            u, s, v = svds(self.utility_matrix, k=self.n_latent_factors)
            sv_matrix = np.diag(s)
            self.Q = u
            self.P = sv_matrix @ v
            self.P = self.P.T
        else:
            self.Q, self.P = None, None

        self.Bu = user_averages
        self.Bi = item_averages
        self.train_dict = train_dict
        self.mean_rating = global_mean
        self._optimize(num_epochs)

        return self

    def _optimize(self, num_epochs: int) -> None:
        curr_sse = self._calculate_sse()
        print(f'Epoch: 00  SSE: {int(curr_sse):,}  RMSE: {(curr_sse / len(self.train_dict)) ** .5}')
        for epoch in range(1, num_epochs + 1):
            self._sgd(epoch-1)
            curr_sse = self._calculate_sse()
            if epoch > 9:
                print(f'Epoch: {epoch}  SSE: {int(curr_sse):,}  RMSE: {(curr_sse / len(self.train_dict)) ** .5}')
            else:
                print(f'Epoch: 0{epoch}  SSE: {int(curr_sse):,}  RMSE: {(curr_sse / len(self.train_dict)) ** .5}')

    def _sgd(self, epoch) -> None:
        for (u, b), rating in sorted(self.train_dict.items(), key=lambda _: random()):
            E_ub = 2 * (rating - self.predict(u, b))
            temp_p = 0
            if self.mode == 'Latent':
                temp_p = self.P[u]
                self.P[u] += (.95 ** epoch) * self.eta_p * (E_ub * self.Q[b] - 2 * self.reg_p * self.P[u])

            self.Bu[u] += (.95 ** epoch) * self.eta_bu * (E_ub - 2 * self.reg_bu * self.Bu[u])

            # update Q and its bias
            if self.mode == 'Latent':
                self.Q[b] += (.95 ** epoch) * self.eta_q * (E_ub * temp_p - 2 * self.reg_q * self.Q[b])
            self.Bi[b] += (.95 ** epoch) * self.eta_bi * (E_ub - 2 * self.reg_bu * self.Bi[b])

    def _calculate_sse(self) -> float:
        sse = 0
        for (u, b), rating in self.train_dict.items():
            sse += (rating - self.predict(u, b)) ** 2
        return sse

    def predict(self, u, b):
        if self.mode == 'Latent':
            return self.mean_rating + (self.Q[b] @ self.P[u]) + self.Bi[b] + self.Bu[u]
        else:
            return self.mean_rating + self.Bi[b] + self.Bu[u]

    def predict_list(self, data):
        predictions = []
        for u, b in data:
            predictions.append(self.predict(u, b))
        return predictions

    def _build_utility_matrix(self):
        um = np.zeros((len(self.items), len(self.users)))
        for b in range(len(self.items)):
            for u in range(len(self.users)):
                if (u, b) in self.train_dict:
                    um[b][u] = self.train_dict[u, b] - self.mean_rating
                else:
                    um[b][u] = 0
        return um


