#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:12:51 2022

@author: roxiesun
"""

from solver1d import Sampler
import argparse

import autograd.numpy as np
from autograd import grad
from autograd.numpy import log, sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal, uniform

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal, norm
from scipy.integrate import cumtrapz, quad
from scipy.signal import savgol_filter


parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-samplesize', default=10, type=int, help='Size of raw samples')
parser.add_argument('-rho', default=20, type=int, help='For gain factor')
parser.add_argument('-kappa', default=1000, type=int, help='For gain factor')
parser.add_argument('-gamma', default=1., type=float, help='For bandwidth')
parser.add_argument('-parts', default=245, type=int, help='Total numer of grid points')
parser.add_argument('-seed', default=231, type=int, help='seed')
pars = parser.parse_args()

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.random.seed(pars.seed)

split_, total_ = 20, 1e6



def mixture(x):
    return 1/3 * multivariate_normal.pdf(x, [-5., -5., -5.], np.array([[4., 5., 0.], [5., 64., 0.], [0., 0., 1.]])) + \
        2/3 * multivariate_normal.pdf(x, [10., 25., 1.], np.array([[1/4, 0., 0.], [0., 1/4, 0.], [0., 0., 1/4]]))
        

def mixture_expand(x, y, z): return mixture([x, y, z])
def logfunction_plot(x): return np.log(mixture(x))

# we want to estimate the marginal (log) density of of y = (x1, x2)

def margin_mixture(x):
    return 1/3 * norm.pdf(x, -5., 8.) + 2/3 * norm.pdf(x, 25., 1/2)
        
#def margin_mixture_expand(x, y): return margin_mixture([x, y])
def marglogfunction_plot(x): return np.log(margin_mixture(x))

# def mixture(x):
#     mu1 = np.array([5., 5., 5.])
#     mu2 = np.array([10., 25., 1.])
#     cov1 = np.array([[4., 5., 0.], [5., 64., 0.], [0., 0., 1.]])
#     cov2 = np.array([[1/4, 0., 0.], [0., 1/4, 0.], [0., 0., 1/4]])
#     rv1 = multivariate_normal(mu1, cov1)
#     rv2 = multivariate_normal(mu2, cov2)
#     return 1/3 * rv1.pdf(x) + 2/3 * rv2.pdf(x)

boundary_ = 30.5
axis_x = np.linspace(-boundary_, boundary_, pars.parts)
#pos = np.dstack((axis_X, axis_Y))



 
logprob_grid = marglogfunction_plot(axis_x)
lower_bound, upper_bound = np.min(logprob_grid), np.max(logprob_grid)

sampler = Sampler(f=mixture, dim0=3, dim=1, boundary=[-boundary_, boundary_], xinit=[0., 10., 0.], \
                  samplesize=pars.samplesize, rho=pars.rho, kappa=pars.kappa, gamma=pars.gamma, parts=pars.parts)

    
warmup = 15000
ccmc_density = np.array([sampler.ccmc_zeta]) # Note: bracket here is necessary for later np.vstack, stack along layers/iters
ccmc_weights = exp(np.array([sampler.ccmc_logG]))
ccmc_store_beta = np.array([sampler.ccmc_store_beta])


for iters in range(int(total_)):
    sampler.ccmc_step(iters)
    if iters > warmup:
        if iters % split_ == 0:
            #note: axis 0: columns, axis 1:rows, or
            #      axis 0: , axis 1: cols, axis 2: rows
            ccmc_density = np.vstack((ccmc_density, np.array([sampler.ccmc_zeta])))
            ccmc_weights = np.vstack((ccmc_weights, exp(np.array([sampler.ccmc_logG]))))
            ccmc_store_beta = np.vstack((ccmc_store_beta, np.array([sampler.ccmc_store_beta])))
        if iters % 2000 == 0:
            fig = plt.figure(figsize=(13, 7))
            ax = fig.add_subplot(2, 1, 1)
            ax.set_title('(a) True v.s. estimated log-density, itr = %d' % (iters), fontsize=16)
            plt.plot(axis_x, logprob_grid, '--', label='True')
            plt.plot(axis_x, sampler.ccmc_logG, label='Est.')
            #plt.plot(axis_x, np.median(np.log(ccmc_weights), axis = 0), label='Post.Median.')
            plt.plot(axis_x, np.mean(np.log(ccmc_weights), axis = 0), label='Post.Mean.')
            ax.set_xlabel('$\lambda$(x)', fontsize=13)
            ax.set_ylabel('logPDF', fontsize=13)
            ax.set_ylim(-25, 5)
            legend = ax.legend(loc='upper left',prop={'size': 11})
            
            ax = fig.add_subplot(2, 1, 2)
            plt.hist(ccmc_store_beta.flatten(), bins= 'auto')
            ax.set_title("(b) Histogram of $\lambda$(x) samples, itr = %d" % (iters), fontsize=16)
            
            plt.tight_layout()
            plt.savefig('/Users/roxiesun/OneDrive - The Chinese University of Hong Kong/2022/purdue/research/ccmc_try/pics1/' + '{:04}'.format((iters - warmup)//1000) + '.png')
            plt.show()
            plt.close(fig)
            

np.argmax(logprob_grid) == np.argmax(np.mean(np.log(ccmc_weights), axis = 0))
          

p0_true, err = quad(margin_mixture, -30.5, 0)

sampler.find_idx(25) # 245   
sampler.find_idx(0)
sum(exp(logprob_grid)[:123])*0.25
sum(np.median(ccmc_weights, axis = 0)[1:123])/sum(np.median(ccmc_weights, axis = 0))

def est_intpol(beta):
    idx = sampler.find_idx(beta)
    u = (beta - (sampler.boundary[0] + (idx - 1)*sampler.div_f)) / sampler.div_f
    return (1-u) * np.mean(ccmc_weights, axis = 0)[idx-1] + u * np.mean(ccmc_weights, axis = 0)[idx]


p_total, err_t = quad(est_intpol, -30.5, 30.5)  
p_est, err1 = quad(est_intpol, -30.5, 0) 
  

#p_est, err1 = quad(est_intpol, -30.5, 0)        # git 20.228....
    
# plt.subplot(1, 1, 1).set_title('(a) Contour of true log-density', fontsize=16)
# plt.contour(axis_X, axis_Y, logprob_grid, 70)

# fig = plt.figure(figsize=(13, 7))
# ax = plt.axes(projection='3d')
# w = ax.plot_wireframe(axis_X, axis_Y, logprob_grid)
# #ax.view_init(25, -140)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('logPDF')
# ax.set_title('Wireframe plot of Gaussian 2D KDE');




# mu1 = np.array([-5., -5.])
# mu2 = np.array([10., 25.])
# cov1 = np.array([[4, 5], [5, 64]])
# cov2 = np.array([[1/4, 0.], [0., 1/4]])
# rv1 = multivariate_normal(mu1, cov1)
# rv2 = multivariate_normal(mu2, cov2)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# ax2.contour(axis_X, axis_Y, np.log(1/3 * rv1.pdf(pos) + 2/3 *rv2.pdf(pos)), 80, alpha = 0.75,colors="Black")


# fig3 = plt.figure(figsize=(13, 7))
# ax3 = plt.axes(projection='3d')
# w = ax3.plot_wireframe(axis_X, axis_Y, np.log(1/3 * rv1.pdf(pos) + 2/3 *rv2.pdf(pos)))
# #ax.view_init(180, 90)
# ax3.set_xlabel('x')
# ax3.set_ylabel('y')
# ax3.set_zlabel('PDF')
# #ax3.set_ylim(-boundary_, -5)
# ax3.set_title('Wireframe plot of Gaussian 2D KDE');


######## for the orinal function, cmc needs energy subregions
# axis_x1 = np.linspace(-boundary_, boundary_, 245)
# axis_y1 = np.linspace(-boundary_, boundary_, 245)
# axis_z1 = np.linspace(-10, 10, 245)
# axis_X1, axis_Y1, axis_Z1 = np.meshgrid(axis_x1, axis_y1, axis_z1)
# pos1 = np.stack((axis_X1, axis_Y1, axis_Z1), axis = -1)
# energy_grid = -logfunction_plot(pos1)
# englower_bound, engupper_bound = np.min(energy_grid) - 1, np.max(energy_grid) + 1
