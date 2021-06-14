#!/usr/bin/env pyhton


__author__ = 'Chao Wu'
__date__ = '05/28/2020'
__version__ = '1.0'


import numpy as np
import pandas as pd
from sympy import symbols, Matrix, lambdify
from openopt import LP, NLP
from constants import R, T
from common_rate_laws import E_expression


def optimize_minimal_driving_force(S, Vss, enzymeInfo, concLB, concUB, fixedConcs, addBnds):
	
	f = np.zeros(S.shape[0] + 1)
	f[0] = -1
	
	S = S * Vss[S.columns]   #! nonlinear pathway 
	A = np.concatenate((np.ones((S.shape[1], 1)), R * T * S.T), axis = 1)
	
	b = -np.array([-R * T * np.log(item[0]) for item in enzymeInfo.loc[:, 'Keq']])
	b = b * Vss[S.columns].values   #! nonlinear pathway

	lb = [-np.inf] + [np.log(concLB)] * S.shape[0]
	ub = [np.inf] + [np.log(concUB)] * S.shape[0]
	
	if addBnds:
		for metab, (lvalue, uvalue) in addBnds.items():
			i = np.where(S.index == metab)[0].item() + 1
			lb[i] = np.log(lvalue)
			ub[i] = np.log(uvalue)
	
	if fixedConcs:
		Aeq = np.zeros((len(fixedConcs), S.shape[0]+1))
		for i, metab in enumerate(fixedConcs.keys()):
			j = np.where(S.index == metab)[0].item() + 1
			Aeq[i, j] = 1
		beq = np.log(list(fixedConcs.values()))
	
		p = LP(f = f, A = A, b = b, Aeq = Aeq, beq = beq, lb = lb, ub = ub, iprint = -1, name = 'Maximize minimal driving force')
	
	else:
		p = LP(f = f, A = A, b = b, lb = lb, ub = ub, iprint = -1, name = 'Maximize minimal driving force')
	
	meth = 'cvxopt_lp'
	r = p.solve(meth, plot = 0)
	
	
	optLogConcs = r.xf[1:]
	optConcs = pd.Series(np.exp(optLogConcs), index = S.index)   #! convert log(concentrations) to concentrations
	
	optDeltaGs = pd.Series(-b + R * T * np.dot(S.T, optLogConcs), index = S.columns)
	refDeltaGs = pd.Series(-b, index = S.columns)
	
	return optConcs, optDeltaGs, refDeltaGs


def optimize_enzyme_cost(S, Vss, enzymeInfo, concLB, concUB, fixedConcs, addBnds):
	
	def get_initial_guess(S, enzymeInfo, concLB, concUB):
		
		nMetabs = len(S.index)
		
		ini = []
		for i, metab in enumerate(S.index):
		
			f = np.zeros(nMetabs)
			f[i] = 1
			
			A = R * T * S.T
			b = -np.array([-R * T * np.log(item[0]) for item in enzymeInfo.loc[:, 'Keq']])
			
			lb = [np.log(concLB)] * nMetabs
			ub = [np.log(concUB)] * nMetabs
			
			meth = 'pclp'
		
			p = LP(f = f, A = A, b = b, lb = lb, ub = ub, iprint = -1)
			r = p.solve(meth, plot = 0)
			
			ini.append(r.xf[i])
			
		ini = np.array(ini)
		
		return ini
	
	
	def get_enzyme_cost(logConcs, S, Vss, enzymeInfo):
		
		enzyCosts = []
		for enzyme in S.columns:
		
			ifRev = enzymeInfo.loc[enzyme, 'rev']
		
			subMasks = S.loc[:, enzyme] < 0
			subLogConcs = logConcs[subMasks]
			subCoes = S.loc[subMasks, enzyme].abs().values
			subKms = np.array([item[0] for item in enzymeInfo.loc[enzyme, 'subsKm'].loc[subMasks]])
			
			proMasks = S.loc[:, enzyme] > 0
			proLogConcs = logConcs[proMasks]
			proCoes = S.loc[proMasks, enzyme].abs().values
			proKms = np.array([item[0] for item in enzymeInfo.loc[enzyme, 'prosKm'].loc[proMasks]])
			
			MW = enzymeInfo.loc[enzyme, 'MW']
			v = Vss[enzyme]
			kcat = enzymeInfo.loc[enzyme, 'kcat'][0]
			deltaGm = -R * T * np.log(enzymeInfo.loc[enzyme, 'Keq'][0])
			
			enzyCost = MW * E_expression(ifRev, subLogConcs, subCoes, subKms, proLogConcs, proCoes, proKms, v, kcat, deltaGm)
			
			enzyCosts.append(enzyCost)
		
		return enzyCosts	
	
	
	def symbolize_f(logConcs, S, Vss, enzymeInfo):
		
		obj = np.sum(get_enzyme_cost(logConcs, S, Vss, enzymeInfo))
		
		return obj
		
		
	def symbolize_dfdx(logConcs, S, Vss, enzymeInfo):
		
		jac = Matrix([symbolize_f(logConcs, S, Vss, enzymeInfo)]).jacobian(logConcs)
		
		return jac
	
		
	logConcs = np.array(symbols(' '.join(S.index)))
	
	pref = lambdify(logConcs, symbolize_f(logConcs, S, Vss, enzymeInfo), modules = 'numpy')	
	predfdx = lambdify(logConcs, symbolize_dfdx(logConcs, S, Vss, enzymeInfo), modules = 'numpy')	
	
	f = lambda x, pref, predfdx: pref(*x)
	dfdx = lambda x, pref, predfdx: predfdx(*x)
	
	A = R * T * S.T
	b = -np.array([-R * T * np.log(item[0]) for item in enzymeInfo.loc[:, 'Keq']])
	
	lb = np.full(len(logConcs), np.log(concLB))
	ub = np.full(len(logConcs), np.log(concUB))
	
	if addBnds:
		for metab, (lvalue, uvalue) in addBnds.items():
			i = np.where(S.index == metab)[0].item()
			lb[i] = np.log(lvalue)
			ub[i] = np.log(uvalue)
			
	ini = np.random.uniform(low = lb, high = ub, size = len(logConcs))		
			
	if fixedConcs:
		Aeq = np.zeros((len(fixedConcs), len(logConcs)))
		for i, metab in enumerate(fixedConcs.keys()):
			j = np.where(S.index == metab)[0].item()
			Aeq[i, j] = 1
		beq = np.log(list(fixedConcs.values()))
		
		p = NLP(f = f, x0 = ini, df = dfdx, args = (pref, predfdx), A = A, b = b, Aeq = Aeq, beq = beq, lb = lb, ub =ub, iprint = -1, name = 'Minimize enzyme cost')
		
	else:
		p = NLP(f = f, x0 = ini, df = dfdx, args = (pref, predfdx), A = A, b = b, lb = lb, ub =ub, iprint = -1, name = 'Minimize enzyme cost')
	
	meth = 'ralg'
	r = p.solve(meth, plot = 0)
	
	optLogConcs = r.xf
	optEnzyCostTotal = r.ff
		
	optConcs = pd.Series(np.exp(optLogConcs), index = S.index)   #! convert log(concentrations) to concentrations
	optEnzyCosts = pd.Series(get_enzyme_cost(optLogConcs, S, Vss, enzymeInfo), index = S.columns)
	
	return optConcs, optEnzyCosts, optEnzyCostTotal
