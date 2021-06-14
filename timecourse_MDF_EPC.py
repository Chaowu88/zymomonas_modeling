#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '05/31/2020'
__version__ = '1.0'


r'''
This script performs timecourse thermodynamics analysis and enzyme protein cost analysis with additional metabolite concentration constraints
This is as upgraded version of the functions in PathPaser (https://github.com/Chaowu88/PathParser)

Usage:
python path\to\timecourse_MDF_EPC.py
'''


PATHPARSER_DIR = r'path\to\PathParser\Scripts'
OUT_DIR = r'output\directory'
RXN_FILE = r'path\to\glucose_utilization_pathway.tsv'   # glucose_utilization_pathway.tsv or xylose_utilization_pathway.tsv
KINETICS_FILE = r'path\to\kinetic_data.tsv'   # generated by substrate_uptake_kinetics.py
SUBSTRATE = 'glucose'   # glucose or xylose


import sys
sys.path.append(PATHPARSER_DIR)
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from parse_network import parse_network, get_full_stoichiometric_matrix, get_steady_state_net_fluxes
from optimization import optimize_minimal_driving_force, optimize_enzyme_cost


def plot_and_save_timecourse_MDFs_and_EPCs(out_dir, MDFs, EPCs):
	
	nEnzymes = MDFs.columns.size
	nCols = 4
	nRows = int(nEnzymes/nCols) + 1

	for specie, data, color, title in zip(['MDF', 'EPC'], [MDFs, EPCs], ['#E24A33', '#1f77b4'], 
									    ["Optimized driving force $\Delta$G' (kJ mol$^{-1}$)", 
									     'Enzyme protein cost (g/(mol s$^{-1}$))']):
		
		fig = plt.figure(figsize = (16, nRows*3))
		
		for idx, enzyme in enumerate(data.columns):
			x = data.index
			y = data[enzyme]
		
			ax = plt.subplot(nRows, nCols, idx+1)
			
			ax.plot(x, y, color = color, linewidth = 4)
			
			ax.ticklabel_format(useOffset = False)
			ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
			
			if abs((y.max() - y.min()) / y.mean()) < 0.1:
				ax.set_ylim(*sorted([y.mean()*0.8, y.mean()*1.2]))
			
			ax.tick_params(labelsize = 20)
			if idx + 4 > nEnzymes - 1:
				ax.set_xlabel('Time (h)', fontsize = 25)
			ax.set_ylabel(enzyme, fontsize = 25)
		
		fig.suptitle(title, y = 1.05, fontsize = 30) 
		
		fig.tight_layout()
		fig.savefig('%s/timecourse_%s.jpg' % (out_dir, specie), dpi = 300, bbox_inches = 'tight')
		plt.close()
		
		data.to_csv('%s/timecourse_%s.tsv' % (out_dir, specie), sep = '\t')

	
	

if __name__ == '__main__':

	if SUBSTRATE == 'glucose':
		colName = 'glc_c'
		metabName = 'Glcex'
		addBnds = {'Glcex': [0.001, 1000]} 
		speEnz, speFlux = 'glf', 1

	elif SUBSTRATE == 'xylose':
		colName = 'xyl_c'
		metabName = 'Xylex'
		addBnds = {'Xylex': [0.001, 1000]} 
		speEnz, speFlux = 'glf', 1.2
	
	exMetabs = ['ATP', 'ADP', 'Pi', 'NADH', 'NAD', 'NADPH', 'NADP']
	concLB, concUB = [0.001, 10]
	
	os.makedirs(OUT_DIR, exist_ok = True)
	
	# parse network
	S4Bal, S4Opt, enzymeInfo, metabInfo = parse_network(RXN_FILE, [], [], exMetabs, [])
	S4BalFull = get_full_stoichiometric_matrix(S4Bal, metabInfo)
	Vss = get_steady_state_net_fluxes(S4BalFull, enzymeInfo, metabInfo, speEnz, speFlux)
	
	# estimate timecourse MDF and EPC
	kineticData = pd.read_csv(KINETICS_FILE, sep = '\t', index_col = 0)
	
	MDFs = pd.DataFrame(index = kineticData.index, columns = enzymeInfo.index)
	EPCs = pd.DataFrame(index = kineticData.index, columns = enzymeInfo.index)
	for time, conc in kineticData[colName].iteritems():
		
		if conc > 0.002:
			fixedConcs = {metabName: conc}
			addBnds = {metabName: [0.001, 1000]}  
			
			optConcs, optDeltaGs, refDeltaGs = optimize_minimal_driving_force(S4Opt, Vss, enzymeInfo, concLB, concUB, fixedConcs, addBnds)
			optConcs, optEnzyCosts, optEnzyCostTotal = optimize_enzyme_cost(S4Opt, Vss, enzymeInfo, concLB, concUB, fixedConcs, addBnds)
		
		try:
			MDFs.loc[time, :] = optDeltaGs
			EPCs.loc[time, :] = optEnzyCosts*1000
		
		except:
			raise ValueError('substrate concentration too low')
	
	plot_and_save_timecourse_MDFs_and_EPCs(OUT_DIR, MDFs, EPCs)
	
