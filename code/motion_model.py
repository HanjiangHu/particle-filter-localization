'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    Updated by Hanjiang Hu (weidong@andrew.cmu.edu), 2022
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0005
        self._alpha2 = 0.0005
        self._alpha3 = 0.001
        self._alpha4 = 0.001

    def WrapToPi(self, angle):
        angWrap = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
        return angWrap


    def sample_array(self, mu, sigma, size):
        return np.random.normal(mu, sigma, size)

    def update_array(self, u_t0, u_t1, x_t0):
        if u_t1[0] == u_t0[0] and u_t1[1] == u_t0[1] and u_t1[2] == u_t0[2]:
            x_t1 = x_t0
            return x_t1

        x_t1 = np.zeros(x_t0.shape)

        deltaR1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        deltaR1 = self.WrapToPi(deltaR1)
        deltaTrans = np.sqrt((u_t1[0] - u_t0[0]) ** 2 + (u_t1[1] - u_t0[1]) ** 2)
        deltaR2 = u_t1[2] - u_t0[2] - deltaR1
        deltaR2 = self.WrapToPi(deltaR2)

        Rot1 = deltaR1 - self.sample_array(0, self._alpha1 * deltaR1 ** 2 + \
                                     self._alpha2 * deltaTrans ** 2, x_t1[:, 0].shape)
        Trans = deltaTrans - self.sample_array(0, self._alpha3 * deltaTrans ** 2 + \
                                         self._alpha4 * deltaR1 ** 2 + self._alpha4 * deltaR2 ** 2, x_t1[:, 0].shape)
        Rot2 = deltaR2 - self.sample_array(0, self._alpha1 * deltaR2 ** 2 + \
                                     self._alpha2 * deltaTrans ** 2, x_t1[:, 0].shape)
        Rot1 = self.WrapToPi(Rot1)
        Rot2 = self.WrapToPi(Rot2)

        x_t1[:, 0] = x_t0[:, 0] + Trans * np.cos(x_t0[:, 2] + Rot1)
        x_t1[:, 1] = x_t0[:, 1] + Trans * np.sin(x_t0[:, 2] + Rot1)
        x_t1[:, 2] = x_t0[:, 2] + Rot1 + Rot2
        return x_t1

