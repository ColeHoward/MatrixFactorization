from typing import Tuple

from scipy.sparse.linalg import svds
from collections import defaultdict
from random import random
import numpy as np
from timeit import default_timer as timer



class LFMpp(object):
    def __init__(self, n_latent_factors=20, reg_q=10, reg_p=1, reg_bi=0.001, reg_bu=0.1, reg_r=1,
                 eta_q=.006, eta_p=.006, eta_bi=.006, eta_bu=.006, eta_r=.004,
                 users=None, items=None, mode='Latent'):
        self.reg_q: float = reg_q
        self.reg_p: float = reg_p
        self.reg_r: float = reg_r
        self.reg_bi: float = reg_bi
        self.reg_bu: float = reg_bu
        self.eta_q: float = eta_q
        self.eta_p: float = eta_p
        self.eta_r: float = eta_r
        self.eta_bi: float = eta_bi
        self.eta_bu: float = eta_bu
        self.n_latent_factors: int = n_latent_factors
        self.train_dict: dict = {}
        self.user_to_items: dict = {}
        self.Q: np.ndarray = None
        self.P: np.ndarray = None
        self.R: np.ndarray = None
        self.Bu: np.ndarray = None
        self.Bi: np.ndarray = None
        self.mean_rating: float = 0.0
        self.users: set = users
        self.items: set = items
        self.utility_matrix: np.ndarray = self.build_utility_matrix()


    def fit(self, train_dict, user_to_items, bu, bi, global_mean, num_epochs=10):
        np.random.seed(1)
        u, s, v = svds(self.utility_matrix, k=self.n_latent_factors)
        sv_matrix = np.diag(s)
        self.Q = u
        self.P = sv_matrix @ v
        self.P = self.P.T
        self.R = np.zeros(self.Q.shape)
        self.Bu = bu
        self.Bi = bi
        self.train_dict = train_dict
        self.user_to_items = user_to_items
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
            shrink = 1
            R_u = np.zeros(self.n_latent_factors)
            rated_items = self.user_to_items[u]
            for bid in rated_items:
                R_u += self.R[bid]
            R_u = R_u / (len(rated_items) ** .5)

            E_ub = 2 * (rating - self.predict(u, b))

            # update P and its bias
            self.P[u] += (.95 ** epoch) * self.eta_p * (E_ub * self.Q[b] - 2 * self.reg_p * shrink * self.P[u])
            self.Bu[u] += (.95 ** epoch) * self.eta_bu * (E_ub - 2 * self.reg_bu * shrink * self.Bu[u])

            # update Q and its bias
            self.Q[b] += (.95 ** epoch) * self.eta_q * (E_ub * (self.P[u] + R_u) - 2 * self.reg_q * self.Q[b])
            self.Bi[b] += (.95 ** epoch) * self.eta_bi * (E_ub - 2 * self.reg_bu * self.Bi[b])

            scaler = self.Q[b] / (len(rated_items) ** .5)
            for bid in rated_items:
                self.R[bid] += (.95 ** epoch) * self.eta_r * E_ub * scaler - 2 * self.reg_r * self.R[bid]


    def _calculate_sse(self) -> float:
        sse = 0
        for (u, b), rating in self.train_dict.items():
            sse += (rating - self.predict(u, b)) ** 2

        return sse

    def predict(self, u, b) -> float:
        return self.mean_rating + self.Bi[b] + self.Bu[u] + self.Q[b] @ (self.P[u] + self.R[b])

    def predict_list(self, data) -> list:
        predictions = []
        for u, b in data:
            predictions.append(self.predict(u, b))
        return predictions

    def _get_averages(self) -> Tuple[np.ndarray, np.ndarray]:
        user_to_ratings, item_to_ratings = defaultdict(list), defaultdict(list)
        user_avgs = np.zeros(len(self.users))
        item_avgs = np.zeros(len(self.items))

        for (u, i), rating in self.train_dict.items():
            user_to_ratings[u].append(rating)
            item_to_ratings[i].append(rating)

        for u, vals in user_to_ratings.items():
            user_avgs[u] = (sum(vals) / len(vals)) - self.mean_rating
        for i, vals in item_to_ratings.items():
            item_avgs[b] = (sum(vals) / len(vals)) - self.mean_rating

        return user_avgs, item_avgs

    def build_utility_matrix(self) -> np.ndarray:
        um = np.zeros((len(self.items), len(self.users)))
        for b in range(len(self.items)):
            for u in range(len(self.users)):
                if (u, b) in self.train_dict:
                    um[b][u] = self.train_dict[u, b] - self.mean_rating
                else:
                    um[b][u] = 0
        return um






