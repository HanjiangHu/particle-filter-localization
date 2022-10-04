'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    Updated by Hanjiang Hu (weidong@andrew.cmu.edu), 2022
'''


from tqdm import tqdm
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader




class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 300  
        self._z_short = 35 
        self._z_max = 30  
        self._z_rand = 200  
        self._sigma_hit = 100
        self._lambda_short = 15000
        self._min_probability = 0.35
        self._subsampling = 2

        """ Occupancy map specs """
        self.OccMap = occupancy_map
        self.OccMapSize = np.size(occupancy_map)
        self.resolution = 10

        """ Laser specs """
        self.laserMax = 8183  # Laser max range
        self.nLaser = 30
        self.laserX = np.zeros((self.nLaser, 1))
        self.laserY = np.zeros((self.nLaser, 1))
        self.beamsRange = np.zeros((self.nLaser, 1))


    def WrapToPi(self, angle):
        angle_wrapped = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
        return angle_wrapped


    def getProbability_array(self, z_star, z_reading):
        # hit
        assert z_star.shape == z_reading.shape, "z_star and z_reading shape not matching"
        reading_ge_0 = np.where(z_reading >= 0, True, False)
        reading_le_max = np.where(z_reading <= self.laserMax, True, False)
        hit_index = np.argwhere(reading_ge_0 & reading_le_max)
        pHit = np.zeros(z_star.shape)
        pHit[hit_index[:, 0], hit_index[:, 1]] = (np.exp(-1 / 2 * (z_reading[hit_index[:, 0], hit_index[:, 1]] - z_star[hit_index[:, 0], hit_index[:, 1]]) ** 2 / (self._sigma_hit ** 2))) / (np.sqrt(2 * np.pi * self._sigma_hit ** 2))

        # short
        reading_le_star = np.where(z_reading <= z_star, True, False)
        short_index = np.argwhere(reading_ge_0 & reading_le_star)
        pShort = np.zeros(z_star.shape)
        eta = 1
        pShort[short_index[:, 0], short_index[:, 1]] = eta * self._lambda_short * np.exp(-self._lambda_short * z_reading[short_index[:, 0], short_index[:, 1]])

        # max
        reading_ge_max = np.where(z_reading >= self.laserMax, True, False)
        max_index = np.argwhere(reading_ge_max)
        pMax = np.zeros(z_star.shape)
        pMax[max_index[:, 0], max_index[:, 1]] = self.laserMax


        # rand
        reading_lt_max = np.where(z_reading < self.laserMax, True, False)
        rand_index = np.argwhere(reading_ge_0 & reading_lt_max)
        pRand = np.zeros(z_star.shape)
        pRand[rand_index[:, 0], rand_index[:, 1]] = 1 / self.laserMax

        p = self._z_hit * pHit + self._z_short * pShort + self._z_max * pMax + self._z_rand * pRand
        p /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
        return p

    def rayCast(self, x_t1):

        '''
         vectorizing (mask) ---
        '''
        

        beamsRange = np.zeros((x_t1.shape[0], self.nLaser))

        L = 25

        xc = x_t1[:, 0]
        yc = x_t1[:, 1]
        myPhi = x_t1[:, 2]
        ang = myPhi - np.pi / 2
        ang = self.WrapToPi(ang)
        offSetX = xc + L * np.cos(ang)
        offSetY = yc + L * np.sin(ang)

        angStep = np.pi / self.nLaser

 
        r = np.linspace(self.laserMax, 0,  800)

        # num of particles * nLaser
        ang_all = np.tile(ang, (self.nLaser, 1)).transpose() + np.tile(np.linspace(0, np.pi, self.nLaser)[0:], (x_t1.shape[0], 1))

        ang_all = self.WrapToPi(ang_all)
        # num of particles * n_r
        r_all = np.tile(r, (x_t1.shape[0], 1))

        x = np.matmul(np.expand_dims(np.cos(ang_all), axis=2), np.expand_dims(r_all, axis=1)) + np.expand_dims(np.expand_dims(offSetX, axis=1), axis=1)
        y = np.matmul(np.expand_dims(np.sin(ang_all), axis=2), np.expand_dims(r_all, axis=1)) + np.expand_dims(np.expand_dims(offSetY, axis=1), axis=1)

        xInt = np.floor(x / self.resolution).astype(int)
        yInt = np.floor(y / self.resolution).astype(int)



        X_cond = np.where(xInt < 800, True, False)
        Y_cond = np.where(yInt < 800, True, False)
        map_cond = np.where(np.abs(self.OccMap[yInt % 800, xInt % 800]) > self._min_probability, True, False)

        cond = X_cond & Y_cond & map_cond
        cond_index = np.argwhere(cond)
        beamsRange[cond_index[:, 0], cond_index[:, 1]] = r[cond_index[:, 2]]

        
        return beamsRange 


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        # print("\n ---------\nBEAM RANGE FINDER MODEL CALLED\n ---------\n")

        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        
        q = 0

        step = int(180 / self.nLaser)
        z_reading = [z_t1_arr[n] for n in range(0, 180, step)]
        z_reading = np.tile(np.array(z_reading), (x_t1.shape[0], 1))

        zt_star = self.rayCast(x_t1)
        p = self.getProbability_array(zt_star, z_reading)
        q = np.sum(np.log(p), axis=1)

        q = self.nLaser / np.abs(q)


        xInt = (x_t1[:, 0] / self.resolution).astype(int)
        yInt = (x_t1[:, 1] / self.resolution).astype(int)
        wrong_index = np.argwhere(np.abs(self.OccMap[yInt, xInt]) ==1.0)
        q[wrong_index] = 0
        return q 