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
    def __init__(self, f=None,dim0=None, dim=None, boundary=None, xinit=None, partition=None, samplesize=10, rho=20, kappa=1000, parts = 244):
        
        self.f = f
        
        self.dim0 = dim0
        self.dim = dim
        self.boundary = boundary #boundary of the lattice ?
        self.xinit = np.array(xinit)
        self.partition = partition
        #self.partition_eng = partition_eng # for the cmc methods
        #self.partition_lmargp = partition_lmargp # for the ccmc kde
        
        self.samplesize = samplesize
        self.rho = rho
        self.kappa =  kappa
        self.parts = parts # the m subregions of cmc
        
        
        
        # initialization for ccmc,
        self.samc_beta = self.xinit[1]
        self.samc_logG = np.zeros(self.parts)#np.log(np.ones(self.parts) / self.parts)
        self.div_f = (self.partition[1] - self.partition[0]) / self.parts
        #self.ccmc_I = self.parts - 1
        self.samc_J = self.parts # because now self.parts = 244 for cmc
        self.samc_rawsample = self.xinit
        self.samc_store_rawsample = self.xinit + np.random.multivariate_normal(np.zeros(3), (4 ** 3) * np.eye(3), size = self.samplesize)#np.empty([self.samplesize, self.dim0])
        self.samc_store_beta = np.array(self.samc_store_rawsample[:,1])
        self.samc_rgcounts = np.zeros(self.parts)
        
    # define for a single sample
        
    def in_domain(self, beta): return (not (beta < self.boundary[0] or beta > self.boundary[1]))

    # note, it returns 1 ~ m-1, where m is the number of subregions?
    #def find_idx(self, beta): return(min(max(int((-log(self.f(beta)) - self.partition[0]) / self.div_f + 1), 1), self.parts))
    def find_idx(self, beta): return(min(max(int((beta - self.partition[0]) / self.div_f + 1), 1), self.parts))
    
    
    
    
    def samc_step(self, iters):
        
        #sampling step
        self.samc_rgcounts = np.zeros(self.parts)
        s = 0
        while s < self.samplesize:
            proposal = self.samc_store_rawsample[s] + np.random.multivariate_normal(np.zeros(3), (5 ** 3) * np.eye(3))
            if self.in_domain(proposal[1]):
                prl_idx = self.find_idx(proposal[1])
                crt_idx = self.find_idx(self.samc_store_beta[s])
                #ratio = exp(log(self.f(proposal)) - log(self.intpol(proposal[1])) - log(self.f(self.ccmc_rawsample)) + log(self.intpol(self.ccmc_beta)))
                ratio = (self.f(proposal) * exp(self.samc_logG[crt_idx-1])) / (exp(self.samc_logG[prl_idx-1]) * self.f(self.samc_store_rawsample[s]))
                if min(ratio, 1) > np.random.uniform():
                    self.samc_rawsample = proposal
                    self.samc_beta = proposal[1]
                    self.samc_store_rawsample[s] = proposal # sth row vector
                    self.samc_store_beta[s] = proposal[1]
                    self.samc_rgcounts[prl_idx-1] += (1 / self.samplesize)
                else:
                    self.samc_rgcounts[crt_idx-1] += (1 / self.samplesize)
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
        
        
        #update the working estimate logG
        self.samc_logG = self.samc_logG + delta * (self.samc_rgcounts - 1 / self.parts)
        self.samc_logG = log(exp(self.samc_logG)/sum(exp(self.samc_logG))*4)
        
        
                
                  
        
        
        