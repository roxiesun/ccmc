#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:12:51 2022

@author: roxiesun
"""

from solver_cmc0913 import Sampler
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
parser.add_argument('-parts', default=244, type=int, help='Total numer of grid points')
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
def logfunction_plot(x): return np.log(mixture(x)) # energy function is the negative of this

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
data=np.concatenate((np.random.normal(-5, 8., 500000), np.random.normal(25, .5, 250000)))


 
logprob_grid = marglogfunction_plot(axis_x)
#lower_bound, upper_bound = np.min(-logprob_grid), np.max(-logprob_grid)


sampler = Sampler(f=mixture, dim0=3, dim=1, boundary=[-boundary_, boundary_], xinit=[0., 10., 0.], \
                  partition = [-boundary_, boundary_], samplesize=pars.samplesize, rho=pars.rho, kappa=pars.kappa, parts=pars.parts)

    
warmup = 10000
# Note: bracket here is necessary for later np.vstack, stack along layers/iters
samc_hist = exp(np.array([sampler.samc_logG]))
samc_store_beta = np.array([sampler.samc_store_beta])
samc_store_cnt = np.array([sampler.samc_rgcounts])


for iters in range(int(total_)):
    sampler.samc_step(iters)
    if iters > warmup:
        if iters % split_ == 0:
            #note: axis 0: columns, axis 1:rows, or
            #      axis 0: , axis 1: cols, axis 2: rows
            samc_store_cnt = np.vstack((samc_store_cnt, np.array([sampler.samc_rgcounts])))
            samc_hist = np.vstack((samc_hist, exp(np.array([sampler.samc_logG]))))
            samc_store_beta = np.vstack((samc_store_beta, np.array([sampler.samc_store_beta])))
        if iters % 2000 == 0:
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(3, 1, 1)
            ax.set_title("(a) True vs fitted histogram of $\lambda$(x),itr = %d" % (iters), fontsize=16)
            plt.hist(data, 244, alpha=.8,range = [-boundary_, boundary_], density = True, label = "true")
            plt.bar(axis_x, np.mean(samc_hist, axis =0), alpha = 0.4, color = 'orange', label = "fitted")
            #plt.hist(samc_store_beta.flatten(), bins=244,alpha=.3,range = [-boundary_, boundary_], density = True, label = "fitted")
            #plt.bar(axis_x, np.median(samc_hist, axis =0), alpha = 0.3, label = "fitted")
            ax.set_xlabel('$\lambda$(x)', fontsize=13)
            plt.legend(loc="upper left", prop={'size': 13})
            
            ax = fig.add_subplot(3, 1, 2)
            ax.set_title("(b) Fitted histogram of $\lambda$(x), itr = %d" % (iters), fontsize=16)
            plt.bar(axis_x, np.mean(samc_hist, axis =0), alpha = 0.8, label = "fitted")
            ax.set_xlabel('$\lambda$(x)', fontsize=13)
            plt.legend(loc="upper left", prop={'size': 13})
            
            
            
            ax = fig.add_subplot(3, 1, 3)
            ax.set_title("(c) Histogram of $\lambda$(x) sampled by CMC, itr = %d" % (iters), fontsize=16)
            plt.hist(samc_store_beta.flatten(), bins=244,alpha=.8,range = [-boundary_, boundary_], density = True)
            ax.set_xlabel('$\lambda$(x)', fontsize=13)
            

            
            plt.tight_layout()
            plt.savefig('/Users/roxiesun/OneDrive - The Chinese University of Hong Kong/2022/purdue/research/samc_try/pics1/' + '{:08}'.format((iters - warmup)//1000) + '.png')
            plt.show()
            plt.close(fig)
            


          



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
