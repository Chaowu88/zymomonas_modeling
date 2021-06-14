#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '05/25/2020'
__version__ = '1.0'


r'''
This script estimates the kinetic parameters of glucose facilitator protein (glf) which transports both glucose and xylose with  competitive inhibition of each other

Example:
python C:\Users\cwu\Desktop\Software\Papers\Zymomonas\github\substrate_uptake_kinetics.py
'''


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import least_squares
from openopt import NLP
import matplotlib.pyplot as plt


OUT_DIR = r'C:\Users\cwu\Desktop\Software\Papers\Zymomonas\github'
DATA_File = r'C:\Users\cwu\Desktop\Software\Papers\Zymomonas\github\data.xlsx'

OD2BIOMASS = 0.33
GLC_MW = 180.156	# g/mol
XYL_MW = 150.13		# g/mol


class ParamsFitting:
	
	def __init__(self, data_file):
	
		self.data = pd.read_excel(DATA_File, index_col = 0, usecols = [0, 1, 2, 3], names = ['time', 'OD', 'glucose', 'xylose'])
		
		self.timepoints = self.data.index.values
		self.exp_biom = (self.data['OD'] * OD2BIOMASS).values			# g/L
		self.exp_glc = (self.data['glucose'] / GLC_MW * 1000).values	# mmol/L
		self.exp_xyl = (self.data['xylose'] / XYL_MW * 1000).values		# mmol/L
		self.ini_concs = [self.exp_glc[0], self.exp_xyl[0]]
		
		self.interp_biomass = interp1d(self.timepoints, self.exp_biom, kind = 'linear', 
									   fill_value = 'extrapolate')   # don't use cubic, or biomass will be negative
									   
		
	@staticmethod
	def v_glucose(c_glc, c_xyl, vmax_glc, km_glc, ki_xyl):
	
		return vmax_glc * c_glc / (km_glc * (1 + c_xyl/ki_xyl) + c_glc)	
		
	
	@staticmethod
	def v_xylose(c_xyl, c_glc, vmax_xyl, km_xyl, ki_glc):
	
		return vmax_xyl * c_xyl / (km_xyl * (1 + c_glc/ki_glc) + c_xyl)
	
	
	def _concs_der(self, t, y, *args):
	
		c_glc, c_xyl = y
		vmax_glc, vmax_xyl, km_glc, km_xyl, ki_glc, ki_xyl = args
		
		biomass = self.interp_biomass(t)
		
		dglcdt = -self.v_glucose(c_glc, c_xyl, vmax_glc, km_glc, ki_xyl) * biomass
		dxyldt = -self.v_xylose(c_xyl, c_glc, vmax_xyl, km_xyl, ki_glc) * biomass
		
		return dglcdt, dxyldt
	
	
	def _concs(self, params, ts):
	
		glucose, xylose = odeint(self._concs_der, y0 = self.ini_concs, t = ts, tfirst = True, args = tuple(params)).T
		
		return glucose, xylose
	
	
	def fit(self):
		
		def f(x):
			glucose, xylose = self._concs(x, self.timepoints)
			resid_glucose = glucose - self.exp_glc
			resid_xylose = xylose - self.exp_xyl
			SSR = np.sum(resid_glucose**2) + np.sum(resid_xylose**2)
			return SSR
		
		x0 = [10, 2, 50, 100, 10, 10]
		res = NLP(f, x0 = x0, lb = [0]*len(x0), ub = [np.inf]*len(x0)).solve('ralg')
			
		self.fitted_params = pd.Series(res.xf, index = ['Vmax_glc(mmol/gDCW/L)', 'Vmax_xyl(mmol/gDCW/L)', 
														'Km_glc(mmol/L)', 'Km_xyl(mmol/L)', 'Ki_glc(mmol/L)', 
														'Ki_xyl(mmol/L)'])
		
		sim_glc, sim_xyl = self._concs(self.fitted_params.values, self.timepoints)
		R2 = 1 - (np.sum((self.exp_glc - sim_glc)**2) + np.sum((self.exp_xyl - sim_xyl)**2)) / \
				 (np.sum((self.exp_glc - self.exp_glc.mean())**2) + np.sum((self.exp_xyl - self.exp_xyl.mean())**2)) 
		print('R2 = %.3f' % R2)
		
		
	def save_fitted_parameters(self, out_dir):
		
		if hasattr(self, 'fitted_params'):
			self.fitted_params.to_csv('%s/fitted_params.tsv' % out_dir, sep = '\t', header = False)
		
		else:
			raise AttributeError('run fit method first')
	
	
	def plot_fitted_vs_measured_curve(self, out_dir):
		
		if hasattr(self, 'fitted_params'):
			ts = np.linspace(self.timepoints.min(), self.timepoints.max(), 1000)
			sim_glc, sim_xyl = self._concs(self.fitted_params.values, ts)
			
			plt.plot(ts, sim_glc, 'forestgreen', linewidth = 3)
			plt.plot(ts, sim_xyl, 'royalblue', linewidth = 3)

			plt.plot(self.timepoints, self.exp_glc, 'forestgreen', linestyle = '', marker = '.', markersize = 15)
			plt.plot(self.timepoints, self.exp_xyl, 'royalblue', linestyle = '', marker = '.', markersize = 15)
			plt.tick_params(labelsize = 15)

			plt.plot([], [], 'forestgreen', marker = '.', label = 'Glucose')
			plt.plot([], [], 'royalblue', marker = '.', label = 'Xylose')
			plt.legend(loc = 'center', bbox_to_anchor = (0.75, 0.3), fontsize = 18)

			names = ['$V_{max,glc}$', 
					 '$V_{max,xyl}$', 
					 '$K_{m,glc}$', 
					 '$K_{m,xyl}$', 
					 '$K_{i,glc}$', 
					 '$K_{i,xyl}$']
			units = ['mmol gDCW$^{-1}$ h$^{-1}$', 
					 'mmol gDCW$^{-1}$ h$^{-1}$', 
					 'mmol L$^{-1}$', 
					 'mmol L$^{-1}$', 
					 'mmol L$^{-1}$', 
					 'mmol L$^{-1}$']
			paramsStr = ['%s: %.2f %s' % (name, value, unit) for name, value, unit in zip(names, self.fitted_params.values, units)]
			msg = 'Fitted parameters:\n' + '\n'.join(paramsStr)
			plt.text(0.35, 0.5, msg, transform = plt.gca().transAxes, fontsize = 13.5)

			plt.xlabel('Time (h)', fontsize = 20)
			plt.ylabel('Substrate conc. (mmol $L^{-1}$)', fontsize = 20)

			plt.savefig('%s/fitted_vs_measured_data.jpg' % out_dir, dpi = 300, bbox_inches = 'tight')
			plt.close()

		else:
			raise AttributeError('run fit method first')
			
			
	def save_kinetics_data(self, out_dir):
		
		if hasattr(self, 'fitted_params'):
			ts = np.linspace(self.timepoints.min(), self.timepoints.max(), 100)
			c_glc, c_xyl = self._concs(self.fitted_params.values, ts)
			
			vmax_glc, vmax_xyl, km_glc, km_xyl, ki_glc, ki_xyl = self.fitted_params
			v_glc = self.v_glucose(c_glc, c_xyl, vmax_glc, km_glc, ki_xyl) 
			v_xyl = self.v_xylose(c_xyl, c_glc, vmax_xyl, km_xyl, ki_glc)
			
			kineticData = pd.DataFrame({'time': ts, 
										'glc_c': c_glc,
										'xyl_c': c_xyl, 
										'glc_v': v_glc,
										'xyl_v': v_xyl})
			kineticData.to_csv('%s/kinetic_data.tsv' % out_dir, sep = '\t', index = False)
		
		else:
			raise AttributeError('run fit method first')



	
if __name__ == '__main__':
		
	glfFitting = ParamsFitting(DATA_File)
	glfFitting.fit()
	glfFitting.save_fitted_parameters(OUT_DIR)
	glfFitting.plot_fitted_vs_measured_curve(OUT_DIR)
	glfFitting.save_kinetics_data(OUT_DIR)
		
	




















