#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '06/11/2020'
__version__ = '1.0'


r'''
This script simulates growth of Zymomonas mobilis using dynamic flux balance analysis, and compares performance under various substrate ratios and agitation rates

Usage:
python path\to\dfba.py
'''


OUT_DIR = r'output\directory'
MODEL_FILE = r'path\to\zymo_BDO_deltaPDC.json'
DATA_File = r'path\to\measured_kinetics.xlsx'

OD2BIOMASS = 0.33
GLC_MW = 180.156	# g/mol
XYL_MW = 150.13		# g/mol
GLYC_MW = 92.1		# g/mol
ACTN_MW = 88.1		# g/mol
BDO_MW = 90.1		# g/mol

VMAX_GLC = 29.3  	# mmol/gDCW/h
VMAX_XYL = 3.18		# mmol/gDCW/h
VMAX_O2 = 15		# mmol/gDCW/h
VMAX_BDO = 10		# mmol/gDCW/h
VMAX_ACTN = 10		# mmol/gDCW/h
KM_GLC = 40.21		# mmol/L
KM_XYL = 80.96		# mmol/L
KM_O2 = 0.00124		# mmol/L 
KM_BDO = 5			# mmol/L
KM_ACTN = 5			# mmol/L
KI_GLC = 128.86		# mmol/L
KI_XYL = 23.98		# mmol/L
C_O2_GAS = 0.21		# mmol/L
KLA = 30			# 1/h


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from cobra import Reaction, Metabolite
from cobra.io import load_json_model
from cobra.util import add_lp_feasibility, fix_objective_as_constraint, add_lexicographic_constraints


class GrowSimulation:
	
	def __init__(self, model_file):
		
		self.model = load_json_model(model_file)
	
	
	def read_experimental_data(self, data_file):
		
		self.data = pd.read_excel(data_file, index_col = 0, 
								  names = ['time', 'OD', 'glucose', 'xylose', 'glycerol', 'acetoin', 'BDO'])

		self.timepoints = self.data.index.values
		self.exp_biom = (self.data['OD'] * OD2BIOMASS).values				# g/L
		self.exp_glu = (self.data['glucose'] / GLC_MW * 1000).values		# mmol/L
		self.exp_xyl = (self.data['xylose'] / XYL_MW * 1000).values			# mmol/L
		self.exp_glyc = (self.data['glycerol'] / GLYC_MW * 1000).values		# mmol/L
		self.exp_actn = (self.data['acetoin'] / ACTN_MW * 1000).values		# mmol/L
		self.exp_bdo = (self.data['BDO'] / BDO_MW * 1000).values			# mmol/L
		
		self.plotSetting = {0: ['darkorange', 'Biomass', self.exp_biom],
							1: ['crimson', 'Glycerol', self.exp_glyc],
							2: ['saddlebrown', 'Acetoin', self.exp_actn],
							3: ['blueviolet', 'BDO', self.exp_bdo],
							4: ['forestgreen', 'Glucose', self.exp_glu],
							5: ['royalblue', 'Xylose', self.exp_xyl]}
			   
			   
	@staticmethod
	def v_glucose(c_glc, c_xyl):
		
		return VMAX_GLC * c_glc / (KM_GLC * (1 + c_xyl/KI_XYL) + c_glc)

	
	@staticmethod
	def v_xylose(c_xyl, c_glc):
		
		return VMAX_XYL * c_xyl / (KM_XYL * (1 + c_glc/KI_GLC) + c_xyl)
	
	
	@staticmethod
	def v_o2(c_o2):
		
		return VMAX_O2 * c_o2 / (KM_O2 + c_o2)
	
	
	@staticmethod	
	def v_bdo(c_bdo):
		
		return VMAX_BDO * c_bdo / (KM_BDO + c_bdo)
	
	
	@staticmethod	
	def v_actn(c_actn):
		
		return VMAX_ACTN * c_actn / (KM_ACTN + c_actn)	

	
	def _concs_der(self, t, y, *args):
	
		biomass, c_glyc, c_actn, c_bdo, c_glc, c_xyl, c_o2 = y
		kla, = args
		
		with self.model:
			self.model.reactions.get_by_id('EX_glc__D_e').lower_bound = -self.v_glucose(c_glc, c_xyl)
			self.model.reactions.get_by_id('EX_xyl__D_e').lower_bound = -self.v_xylose(c_xyl, c_glc)
			self.model.reactions.get_by_id('EX_o2_e').lower_bound = -self.v_o2(c_o2)
			self.model.reactions.get_by_id('EX_btd_RR_e').lower_bound = -self.v_bdo(c_bdo)
			self.model.reactions.get_by_id('EX_actn__R_e').lower_bound = -self.v_actn(c_actn)
			self.model.reactions.get_by_id('ATPM').lower_bound = 0
			
			add_lp_feasibility(self.model)
			fix_objective_as_constraint(self.model)
			vs = add_lexicographic_constraints(self.model, ['biomass_degraaf', 'EX_glyc_e', 'EX_actn__R_e', 'EX_btd_RR_e', 'EX_glc__D_e', 'EX_xyl__D_e', 'EX_o2_e'], ['max', 'max', 'max', 'max', 'min', 'min', 'min'])
			
		dMdt = vs * biomass
		dMdt[-1] = dMdt[-1] + kla * (C_O2_GAS - c_o2)   # O2 uptake
		
		return dMdt
		
		
	def simulate_and_plot(self, out_dir):
		
		ts = np.linspace(self.timepoints[0], self.timepoints[-1]-0.5, 100)
		ini_concs = [self.exp_biom[0], 0, 0, 0, self.exp_glu[0], self.exp_xyl[0], C_O2_GAS]
		
		sol = odeint(func = self._concs_der, y0 = ini_concs, t = ts, tfirst = True, args = (KLA,))

		
		fig, axes = plt.subplots(2, 3, figsize = (10, 3), sharex = True)

		axes[0, 0].plot(self.timepoints, self.plotSetting[0][2], color = self.plotSetting[0][0], linestyle = '', marker = '.')
		axes[0, 0].plot(ts, sol[:, 0], color = self.plotSetting[0][0])
		axes[0, 0].plot([], [], color = self.plotSetting[0][0], marker = '.', label = self.plotSetting[0][1])
		axes[0, 0].set_ylabel('Conc. (g $L^{-1}$)')

		axes[0, 1].plot(self.timepoints, self.plotSetting[4][2], color = self.plotSetting[4][0], linestyle = '', marker = '.')
		axes[0, 1].plot(ts, sol[:, 4], color = self.plotSetting[4][0])
		axes[0, 1].plot([], [], color = self.plotSetting[4][0], marker = '.', label = self.plotSetting[4][1])
		axes[0, 1].set_ylabel('Conc. (mmol $L^{-1}$)')

		axes[0, 2].plot(self.timepoints, self.plotSetting[5][2], color = self.plotSetting[5][0], linestyle = '', marker = '.')
		axes[0, 2].plot(ts, sol[:, 5], color = self.plotSetting[5][0])
		axes[0, 2].plot([], [], color = self.plotSetting[5][0], marker = '.', label = self.plotSetting[5][1])
		axes[0, 2].set_ylabel('Conc. (mmol $L^{-1}$)')

		axes[1, 0].plot(self.timepoints, self.plotSetting[1][2], color = self.plotSetting[1][0], linestyle = '', marker = '.')
		axes[1, 0].plot(ts, sol[:, 1], color = self.plotSetting[1][0])
		axes[1, 0].plot([], [], color = self.plotSetting[1][0], marker = '.', label = self.plotSetting[1][1])
		axes[1, 0].set_xlabel('Time (h)')
		axes[1, 0].set_ylabel('Conc. (mmol $L^{-1}$)')

		axes[1, 1].plot(self.timepoints, self.plotSetting[2][2], color = self.plotSetting[2][0], linestyle = '', marker = '.')
		axes[1, 1].plot(ts, sol[:, 2], color = self.plotSetting[2][0])
		axes[1, 1].plot([], [], color = self.plotSetting[2][0], marker = '.', label = self.plotSetting[2][1])
		axes[1, 1].set_xlabel('Time (h)')
		axes[1, 1].set_ylabel('Conc. (mmol $L^{-1}$)')

		axes[1, 2].plot(self.timepoints, self.plotSetting[3][2], color = self.plotSetting[3][0], linestyle = '', marker = '.')
		axes[1, 2].plot(ts, sol[:, 3], color = self.plotSetting[3][0])
		axes[1, 2].plot([], [], color = self.plotSetting[3][0], marker = '.', label = self.plotSetting[3][1])
		axes[1, 2].set_xlabel('Time (h)')
		axes[1, 2].set_ylabel('Conc. (mmol $L^{-1}$)')

		fig.legend(loc = 'center', bbox_to_anchor = (1.05, 0.55))

		fig.tight_layout()
		fig.savefig('%s/simualted_vs_measured_data.jpg' % out_dir, dpi = 300, bbox_inches = 'tight')
		plt.close()
	
	
	def _simulation_plot(self, out_file, ts, data):
		
		fig, axes = plt.subplots(2, 3, figsize = (10, 3), sharex = True)

		for label, [sol, ls] in data.items():
			axes[0, 0].plot(ts, sol[:, 0], color = self.plotSetting[0][0], linestyle = ls)
			axes[0, 1].plot(ts, sol[:, 4], color = self.plotSetting[4][0], linestyle = ls)
			axes[0, 2].plot(ts, sol[:, 5], color = self.plotSetting[5][0], linestyle = ls)
			axes[1, 0].plot(ts, sol[:, 1], color = self.plotSetting[1][0], linestyle = ls)
			axes[1, 1].plot(ts, sol[:, 2], color = self.plotSetting[2][0], linestyle = ls)
			axes[1, 2].plot(ts, sol[:, 3], color = self.plotSetting[3][0], linestyle = ls)
			axes[0, 0].plot([], [], color = 'black', linestyle = ls, label = 'k$_L$a = %s' % label)
		fig.legend(loc = 'center', bbox_to_anchor = (1.18, 0.55))

		p1 = axes[0, 0].plot([], [], color = self.plotSetting[0][0])
		p2 = axes[0, 1].plot([], [], color = self.plotSetting[4][0])
		p3 = axes[0, 2].plot([], [], color = self.plotSetting[5][0])
		p4 = axes[1, 0].plot([], [], color = self.plotSetting[1][0])
		p5 = axes[1, 1].plot([], [], color = self.plotSetting[2][0])
		p6 = axes[1, 2].plot([], [], color = self.plotSetting[3][0])
		legend = fig.legend(handles = [p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]], 
							labels = [self.plotSetting[i][1] for i in [0, 4, 5, 1, 2, 3]],
							loc = 'center', bbox_to_anchor = (1.05, 0.55))
		fig.add_artist(legend)

		axes[0, 0].set_ylabel('Conc. (g $L^{-1}$)')
		axes[0, 1].set_ylabel('Conc. (mmol $L^{-1}$)')
		axes[0, 2].set_ylabel('Conc. (mmol $L^{-1}$)')
		for i in range(3):
			axes[1, i].set_xlabel('Time (h)')
			axes[1, i].set_ylabel('Conc. (mmol $L^{-1}$)')

		fig.tight_layout()
		fig.savefig(out_file, dpi = 300, bbox_inches = 'tight')
		plt.close()
		
		
	def simulate_and_plot_substrate_ratios(self, out_dir, ratios, linestyles = ['-', '--', '-.', ':']):
		
		ts = np.linspace(self.timepoints[0], self.timepoints[-1]-10, 100)
		totalC = self.exp_glu[0] * 6 + self.exp_xyl[0] * 5
		
		sols = {}
		for ratio, ls in zip(ratios, linestyles):
			iniGlc = ratio * totalC / (6 * ratio + 5)
			iniXyl = totalC / (6 * ratio + 5)
			ini_concs = [self.exp_biom[0], 0, 0, 0, iniGlc, iniXyl, C_O2_GAS]
			sol = odeint(func = self._concs_der, y0 = ini_concs, t = ts, tfirst = True, args = (KLA,))
			sols[ratio] = [sol, ls]
			
		self._simulation_plot('%s/substrate_simualtion.jpg' % out_dir, ts, sols)
		
		
	def simulate_and_plot_agitation_rates(self, out_dir, klas, linestyles = ['-', '--', '-.', ':']):

		ts = np.linspace(self.timepoints[0], self.timepoints[-1]-10, 100)
		ini_concs = [self.exp_biom[0], 0, 0, 0, self.exp_glu[0], self.exp_xyl[0], C_O2_GAS]
		
		sols = {}
		for kla, ls in zip(klas, linestyles):
			sol = odeint(func = self._concs_der, y0 = ini_concs, t = ts, tfirst = True, args = (kla,))
			sols[kla] = [sol, ls]
		
		self._simulation_plot('%s/agitation_simulation.jpg' % out_dir, ts, sols)




if __name__ == '__main__':
	
	zymoGrowth = GrowSimulation(MODEL_FILE)
	zymoGrowth.read_experimental_data(DATA_File)
	#zymoGrowth.simulate_and_plot(OUT_DIR)
	zymoGrowth.simulate_and_plot_substrate_ratios(OUT_DIR, [0.5, 1, 2, 4])
	zymoGrowth.simulate_and_plot_agitation_rates(OUT_DIR, [10, 40, 70, 100])

