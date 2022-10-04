'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    Updated by Hanjiang Hu (weidong@andrew.cmu.edu), 2022
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        X_bar_resampled = list()
        M = X_bar.shape[0]
        X_bar[:, 3] = X_bar[:, 3] / np.sum(X_bar[:, 3])
        r = np.random.uniform(0, 1.0 / M)
        c = X_bar[0, 3]
        i = 0
        m = np.arange(1, M+1)
        u = r + (m - 1) * 1 / M

        cum_sum = np.tile(np.cumsum(X_bar[:, 3]), (M, 1))
        u_table = np.tile(u, (M, 1)).transpose()

        result = np.where(cum_sum > u_table, cum_sum - u_table, np.inf)
        X_bar_resampled = X_bar[np.argmin(result, axis=1)]
        return X_bar_resampled
