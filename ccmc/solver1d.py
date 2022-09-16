#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:03:41 2022

@author: roxiesun
"""

import sys

import autograd.numpy as np
from autograd import grad
from autograd.numpy import sqrt, sin, cos, exp, pi, prod, log
from autograd.numpy.random import normal, uniform

from time import time
from math import dist


from scipy.stats import multivariate_normal, norm

class Sampler:
    def __init__(self, f=None,dim0=None, dim=None, boundary=None, xinit=None, samplesize=10, rho=20, kappa=1000, gamma=1., parts = 245):
        
        self.f = f
        
        self.dim0 = dim0
        self.dim = dim
        self.boundary = boundary #boundary of the lattice ?
        self.xinit = np.array(xinit)
        #self.partition_eng = partition_eng # for the cmc methods
        #self.partition_lmargp = partition_lmargp # for the ccmc kde
        
        self.samplesize = samplesize
        self.rho = rho
        self.kappa =  kappa
        self.gamma = gamma
        self.parts = parts # the m subregions of cmc
        
        
        
        # initialization for ccmc,
        self.ccmc_beta = self.xinit[1]
        self.ccmc_logG = np.log(np.ones(self.parts) / self.parts)
        self.ccmc_zeta = np.ones(self.parts) / self.parts
        self.div_f = (self.boundary[1] - self.boundary[0])/(self.parts-1)
        #self.ccmc_I = self.parts - 1
        #self.ccmc_J = self.parts - 1
        self.ccmc_rawsample = self.xinit
        self.ccmc_store_rawsample = self.xinit + np.random.multivariate_normal(np.zeros(3), (4 ** 3) * np.eye(3), size = self.samplesize)#np.empty([self.samplesize, self.dim0])
        self.ccmc_store_beta = np.array(self.ccmc_store_rawsample[:,1])
        self.accept_count = np.zeros(self.samplesize)
        
        
    # define for a single sample
        
    def in_domain(self, beta): return (not (beta < self.boundary[0] or beta > self.boundary[1]))

    # note, it returns 1 ~ 244
    def find_idx(self, beta): return(min(max(int((beta - self.boundary[0]) / self.div_f + 1), 1), self.parts - 1))
   
    
   
    def intpol(self, beta):
        idx = self.find_idx(beta)
        u = (beta - (self.boundary[0] + (idx - 1)*self.div_f)) / self.div_f
        return (1-u) * exp(self.ccmc_logG[idx-1]) + u * exp(self.ccmc_logG[idx])
    
    
    def lattice_refine(self, new_parts):
        self.parts = new_parts
        self.ccmc_logG = np.zeros(self.parts)
        self.ccmc_zeta = np.ones(self.parts) / self.parts
        self.div_f = (self.boundary[1] - self.boundary[0])/(self.parts-1)
    
    
    
    
    def ccmc_step(self, iters):
        
        #sampling step
        s = 0
        while s < self.samplesize:
            #proposal = self.ccmc_rawsample + np.random.multivariate_normal(np.zeros(3), (5 ** 3) * np.eye(3))
            proposal = self.ccmc_store_rawsample[s] + np.random.multivariate_normal(np.zeros(3), (5 ** 3) * np.eye(3))
            if self.in_domain(proposal[1]):
                #ratio = exp(log(self.f(proposal)) - log(self.intpol(proposal[1])) - log(self.f(self.ccmc_rawsample)) + log(self.intpol(self.ccmc_beta)))
                #ratio = (self.f(proposal) * self.intpol(self.ccmc_beta)) / (self.intpol(proposal[1]) * self.f(self.ccmc_rawsample))
                ratio = (self.f(proposal) * self.intpol(self.ccmc_store_beta[s])) / (self.intpol(proposal[1]) * self.f(self.ccmc_store_rawsample[s]))
                if min(ratio, 1) > np.random.uniform():
                    self.accept_count[s] += 1
                    self.ccmc_rawsample = proposal
                    self.ccmc_beta = proposal[1]
                    self.ccmc_store_rawsample[s] = proposal # sth row vector
                    self.ccmc_store_beta[s] = proposal[1]
                s += 1
        
        # proposal = self.ccmc_store_rawsample + np.random.multivariate_normal(np.zeros(3), (5 ** 3) * np.eye(3), size = self.samplesize)
        
        # while sum(map(lambda i: self.in_domain(proposal[i,1]), range(self.samplesize))) < self.samplesize :
        #     proposal = self.ccmc_store_rawsample + np.random.multivariate_normal(np.zeros(3), (5 ** 3) * np.eye(3), size = self.samplesize)
        # # its now an array of shape self.samplesize x self.dim0 
        # #if sum(map(lambda i: self.in_domain(proposal[i,1]), range(self.samplesize))) == self.samplesize :
        # ratio = np.array(list(map(lambda i: (self.f(proposal[i,:]) * self.intpol(self.ccmc_store_beta[i])) / (self.intpol(proposal[i,1]) * self.f(self.ccmc_store_rawsample[i])), range(self.samplesize))))
        # for s in [i for i, x in enumerate(np.minimum(ratio,1).reshape(self.samplesize) > uniform(size = self.samplesize)) if x]:
        #     self.ccmc_store_rawsample[s] = proposal[s]
        #     self.ccmc_store_beta[s] = proposal[s,1]
        # estimate updating
        
        # estimate density of the transformed samples y_1, ... y_M by kernel method
        delta = self.rho * self.kappa / max(self.kappa, iters)
        
        #bandwidth = np.diag([min(delta ** self.gamma, np.ptp(self.ccmc_store_beta[:,0]) / (2 * (1 + np.log2(self.samplesize)))), min(delta ** self.gamma, np.ptp(self.ccmc_store_beta[:,1]) / (2 * (1 + np.log2(self.samplesize))))])
        bandwidth = min(delta ** self.gamma, (self.boundary[1] - self.boundary[0]) / (2 * (1 + np.log2(self.samplesize)))) ** 2       
        
        #bandwidth = min(delta ** self.gamma, np.ptp(self.ccmc_store_beta) / (2 * (1 + np.log2(self.samplesize)))) ** 2       
        
        radius = 4*(sqrt(bandwidth))
        # if we consider fast computation:
        sep = int(radius/self.div_f + 1)
        sep_sample_l = np.array(list(map(lambda k: max(1, self.find_idx(self.ccmc_store_beta[k]) - sep), range(self.samplesize))))
        sep_sample_u = np.array(list(map(lambda k: min(self.parts - 1, self.find_idx(self.ccmc_store_beta[k]) + sep), range(self.samplesize))))
        sep_sample = np.column_stack((sep_sample_l, sep_sample_u))
        
        
        for i in range(self.samplesize):
            self.ccmc_zeta[range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)] = 0.
        
        for i in range(self.samplesize):
            self.ccmc_zeta[range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)] += 1/self.samplesize * (bandwidth ** (-1/2)) * norm.pdf((bandwidth ** (-1/2)) * (self.boundary[0] + np.array(range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)) * self.div_f - self.ccmc_store_beta[i]))
        
        #fast computation: evaluate the kernel density at the grid points lying in max(4ht1, 4ht2) of each sample y_k
        #for i in range(self.parts):
            #if sum(map(lambda k: abs(self.boundary[0] + i * self.div_f - self.ccmc_store_beta[k]) <= radius, range(self.samplesize))) > 0 :
            #self.ccmc_zeta[i] = np.mean(np.array(list(map(lambda k: bandwidth ** (-1/2) * norm.pdf(bandwidth ** (-1/2) * (self.boundary[0] + i * self.div_f - self.ccmc_store_beta[k])), range(self.samplesize)))))
        
        # without fast computation, uncomment the line below            
        #self.ccmc_zeta = np.mean(np.array(list(map(lambda k: (bandwidth ** (-1/2)) * norm.pdf((bandwidth ** (-1/2)) * (self.boundary[0] + np.array(range(self.parts)) * self.div_f - self.ccmc_store_beta[k])), range(self.samplesize)))), axis = 0)
        #normalizing
        self.ccmc_zeta = self.ccmc_zeta / np.sum(self.ccmc_zeta)
        
        #update the working estimate logG
        self.ccmc_logG = self.ccmc_logG + delta * (self.ccmc_zeta - (1 / self.parts))
        #if add a constrain s.t. \sum_\xi_{zij} = self.parts/61?
        #self.ccmc_logG = log(exp(self.ccmc_logG)/sum(exp(self.ccmc_logG))*1/self.div_f)
        
        
        
        #check if we need to refine the lattice
        if self.div_f / sqrt(bandwidth) > 8:
            print("Bandwidth too small: d/h is %f \n" % (self.div_f / sqrt(bandwidth)))
            sys.exit('Lattice refining required: ')
            self.lattice_refine(self.parts*2)
                
                  
        
        
        