#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '05/28/2020'
__version__ = '1.0'


r'''
This script performs thermodynamics analysis which maximize the minimal driving force and
					 enzyme protein cost analysis which minimize the totol enzyme protein cost of a given pathway 
					 with additional metabolite concentration constraints
This is as upgraded version of the functions in PathPaser (https://github.com/Chaowu88/PathParser)

Usage:
python path\to\constrained_MDF_and_EPC.py
'''


PATHPARSER_DIR = r'path\to\PathParser\Scripts'
OUT_DIR = r'output\directory'
RXN_FILE = r'path\to\glucose_utilization_pathway.tsv'   # glucose_utilization_pathway.tsv or xylose_utilization_pathway.tsv
SUBSTRATE = 'glucose'   # glucose or xylose


import sys
sys.path.append(PATHPARSER_DIR)
import os
from optimization import optimize_minimal_driving_force, optimize_enzyme_cost
from parse_network import parse_network, get_full_stoichiometric_matrix, get_steady_state_net_fluxes
from output import print_driving_force_optimization_results, plot_cumulative_deltaGs, save_driving_force_optimization_results, print_enzyme_cost_optimization_results, plot_enzyme_costs, save_enzyme_cost_optimization_results




if __name__ == '__main__':
	
	if SUBSTRATE == 'glucose':
		fixedConcs = {'Glcex': 443.0}
		addBnds = {'Glcex': [0.001, 1000]} 
		speEnz, speFlux = 'glf', 1

	elif SUBSTRATE == 'xylose':
		fixedConcs = {'Xylex': 244.5}
		addBnds = {'Xylex': [0.001, 1000]} 
		speEnz, speFlux = 'glf', 1.2
	
	exMetabs = ['ATP', 'ADP', 'Pi', 'NADH', 'NAD', 'NADPH', 'NADP']
	concLB, concUB = [0.001, 10]
	
	os.makedirs(OUT_DIR, exist_ok = True)
	
	# parse network
	S4Bal, S4Opt, enzymeInfo, metabInfo = parse_network(RXN_FILE, [], [], exMetabs, [])
	S4BalFull = get_full_stoichiometric_matrix(S4Bal, metabInfo)
	Vss = get_steady_state_net_fluxes(S4BalFull, enzymeInfo, metabInfo, speEnz, speFlux)
	
	# maximize minimal driving force
	optConcs, optDeltaGs, refDeltaGs = optimize_minimal_driving_force(S4Opt, Vss, enzymeInfo, concLB, concUB, fixedConcs, addBnds)
	
	print_driving_force_optimization_results(optConcs, optDeltaGs, refDeltaGs)
	plot_cumulative_deltaGs(optDeltaGs, refDeltaGs, OUT_DIR)
	save_driving_force_optimization_results(optConcs, optDeltaGs, refDeltaGs, OUT_DIR)

	# minimize enzyme protein cost	
	optConcs, optEnzyCosts, optEnzyCostTotal = optimize_enzyme_cost(S4Opt, Vss, enzymeInfo, concLB, concUB, fixedConcs, addBnds)	
		
	print_enzyme_cost_optimization_results(optConcs, optEnzyCosts, optEnzyCostTotal)
	plot_enzyme_costs(optEnzyCosts, OUT_DIR)
	save_enzyme_cost_optimization_results(optConcs, optEnzyCosts, optEnzyCostTotal, OUT_DIR)
	
