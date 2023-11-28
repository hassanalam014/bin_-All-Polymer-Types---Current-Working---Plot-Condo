# Author: Kier von Konigslow
# Date: May 2017
#
# Description	: The purpose of this file is to estimate the multicomponent fluid parameters
#				  for the PS/CO2 binary mixture.
#

import os,sys,math,csv,numpy as npy
# from p_params import *
# from s_params import *
from loadExperimentalData import *
from lmfit import minimize, Parameters, report_fit
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
# from calculateBinaryResidual import calculateBinaryResidualCHV
from Parameters_of_Different_Polymers import *
from SplittingExperimentalDataOf_X_Sw_aboveANDbelowTgANDIsotherms import *
from Parameters_of_Different_Polymers import *
from Split_Exp_Data_in_Isotherms import*
from calculateBinaryVariablesCHV_Condo_nsolve import * #calculateBinarySolubilitySwellingCondo_Original,binarySolubilitySwellingCondo_Original,binaryPhaseEquilibriumCondo_Original_nsolve, calculateBinarySolubilityCondo_Original, calculateBinarySwellingCondo_Original

def residualFunction(A0,A,weight=1.0):
	if len(A0) != len(A):
		raise ValueError('In residual: The number of experimental points and number of theoretical points are not equal.')
	
	residual = npy.zeros(len(A0))
	
	for i in range(0,len(A0)):
		residual[i] = weight*((A0[i]-A[i]))#/A0[i]  #Kier original had no hash and no absolute
	
	
	# print 'weight is', weight
	return residual

def binaryResidualCondo_Original(P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,**kwargs):

	m_s = npy.zeros(len(P0_X))
	Sw = npy.zeros(len(P0_S))
	res_X = npy.zeros(len(P0_X))
	res_S = npy.zeros(len(P0_S))
	
	if len(P0_X) != len(P0_S):
		warnings.warn('In binaryResidual: Mismatch in solubility and swelling data number. Results may be skewed.')

	suppress_print = kwargs.pop('suppress_print',False)
	if not suppress_print:
		for key,value in kwargs.items():
			print '%s=%s' % (key,value)

	if 'X' in fit_type:
		P0,T0,m_s = calculateBinarySolubilityCondo_Original(P0_X,T0_X,Mp,Ms,**kwargs)
		print 'solubility weight is'
		res_X = residualFunction(X0_X,m_s,1.0-fs)
	if 'S' in fit_type:
		P0,T0,Sw = calculateBinarySwellingCondo_Original(P0_S,T0_S,Mp,Ms,**kwargs)
		# Sw_dash=npy.array(Sw)-1
		# S0_S_dash=npy.array(S0_S)-1
		print 'swelling weight is'
		res_S = residualFunction(S0_S,Sw,fs)
	
	if 'X' in fit_type and 'S' in fit_type:
		# print 'hurry'
		residual = npy.concatenate((res_X,res_S),axis=0)
	elif 'X' in fit_type:
		residual = res_X
	elif 'S' in fit_type:
		residual = res_S
	else:
		raise ValueError('In binaryResidual: fit_type must contain X and/or S.')

	return residual

def calculateBinaryResidualCondo_Original(params,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,fit_type):

	fs = params['fs'].value
	Mp = params['Mp'].value
	Ms = params['Ms'].value
	
	if 'Ppstar' in params and 'Tpstar' in params and 'Rpstar' in params:
		Ppstar = params['Ppstar'].value
		Tpstar = params['Tpstar'].value
		Rpstar = params['Rpstar'].value
		kwargs = {'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar}
	elif 'alpha_p' in params and 'vhp' in params and 'epsilon_p' in params:
		alpha_p = params['alpha_p'].value
		vhp = params['vhp'].value
		epsilon_p = params['epsilon_p'].value
		kwargs = {'alpha_p':alpha_p,'vhp':vhp,'epsilon_p':epsilon_p}
	else:
		raise ValueError('In calculateBinaryResidualCHV, pure polymer: Either molecular (alpha_p,vhp,epsilon_p) or characteristic (Ppstar,Tpstar,Rpstar) parameters must be used.')
	
	if 'Psstar' in params and 'Tsstar' in params and 'Rsstar' in params:
		Psstar = params['Psstar'].value
		Tsstar = params['Tsstar'].value
		Rsstar = params['Rsstar'].value
		kwargs.update({'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar})
	elif 'alpha_s' in params and 'vhs' in params and 'epsilon_s' in params:
		alpha_s = params['alpha_s'].value
		vhs = params['vhs'].value
		epsilon_s = params['epsilon_s'].value
		kwargs.update({'alpha_s':alpha_s,'vhs':vhs,'epsilon_s':epsilon_s})
	else:
		raise ValueError('In calculateBinaryResidualCHV, pure solvent: Either molecular (alpha_s,vhs,epsilon_s) or characteristic (Psstar,Tsstar,Rsstar) parameters must be used.')
	
	if 'zeta' in params:
		zeta = params['zeta'].value
		kwargs.update({'zeta':zeta})
	else:
		raise ValueError('In calculateBinaryResidualCHV, mixture parameters: (k12,delta) or (zeta,delta) mixture parameters must be used.')
	
	if 'verbose' in params:
		verbose = params['verbose'].value
		kwargs.update({'verbose':verbose})
	
	res = binaryResidualCondo_Original(P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,**kwargs)
	
	print '==> Done for values above.'

	return res

Polymer_Type='PS'
Solvent='CO2'

Parameters_Paper ='Self_Grassia'			# P*T*R* and g,epsilon_2,x (PVT-Tg Data Paper or Direct P*T*R* Values Reference)
Cp_Polymer_Weight = '02kilo_POST_THESIS'	# g,epsilon_2,x (Cp Paper Reference)
Paper_Number = 'Paper4_11_12'						# Solubility or Swelling Data Reference
#for PS: 'Paper4_11_12'
#for PMMA: 'Paper15'
kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight}

Ppstar,Tpstar,Rpstar,Mp,Psstar,Tsstar,Rsstar,Ms,P_exp,Tg_exp=Parameters_of_Different_Polymers(**kwargs)
P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete=loadExperimentSwXData(**kwargs)
Far_Above_Data=False
P0_X,P0_X_above_Tg,P0_X_far_above_Tg,T0_X,T0_X_above_Tg,T0_X_far_above_Tg,X0_X,X0_X_above_Tg,X0_X_far_above_Tg,Rubber0_X,Rubber0_X_above_Tg,Rubber0_X_far_above_Tg,P0_S,P0_S_above_Tg,P0_S_far_above_Tg,T0_S,T0_S_above_Tg,T0_S_far_above_Tg,S0_S,S0_S_above_Tg,S0_S_far_above_Tg,Rubber0_S,Rubber0_S_above_Tg,Rubber0_S_far_above_Tg = SplitExperimental_X_Sw_Data(P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete,Far_Above_Data,**kwargs)

number_of_isotherm, result = Split_Isotherms(P0_X,T0_X,X0_X,'X')
P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5,P0_X_T6,T0_X_T6,X0_X_T6,P0_X_T7,T0_X_T7,X0_X_T7,P0_X_T8,T0_X_T8,X0_X_T8 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23]
# print P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5

number_of_isotherm_swelling, result = Split_Isotherms(P0_S,T0_S,S0_S,'S')
P0_S_T1,T0_S_T1,S0_S_T1,P0_S_T2,T0_S_T2,S0_S_T2,P0_S_T3,T0_S_T3,S0_S_T3,P0_S_T4,T0_S_T4,S0_S_T4,P0_S_T5,T0_S_T5,S0_S_T5,P0_S_T6,T0_S_T6,S0_S_T6,P0_S_T7,T0_S_T7,S0_S_T7,P0_S_T8,T0_S_T8,S0_S_T8 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23]
# print P0_S_T1,T0_S_T1,S0_S_T1,P0_S_T2,T0_S_T2,S0_S_T2,P0_S_T3,T0_S_T3,S0_S_T3,P0_S_T4,T0_S_T4,S0_S_T4,P0_S_T5,T0_S_T5,S0_S_T5

# P0_X = P0_X_T2
# T0_X = T0_X_T2
# X0_X = X0_X_T2

# P0_S = P0_S_T2
# T0_S = T0_S_T2
# S0_S = S0_S_T2

print 'P0_X=',P0_X
print 'T0_X=', T0_X
print 'X0_X=', X0_X
print 'Rubber0_X=', Rubber0_X

# P0_X = npy.concatenate((P0_X,[0.0001]),axis=0)
# T0_X = npy.concatenate((T0_X,[373.15]),axis=0)
# X0_X = npy.concatenate((X0_X,[0]),axis=0)

#Initializing the parameters.
params = Parameters()
#The following code sets up the model's parameters. It includes both fitting parameters and parameters that will remain fixed
#for the fitting. The values given are the inital guesses of fitting parameters and values of fixed parameters.
#           	(Name,		Value,		Vary?,	Min,	Max,	Expr)
params.add_many(('zeta',	1.0,		True,	None,	None,	None),
				('Ppstar',	Ppstar,		False,	None,	None,	None),
				('Tpstar',	Tpstar,		False,	None,	None,	None),
				('Rpstar',	Rpstar,		False,	None,	None,	None),
				('Mp',		Mp,			False,	None,	None,	None),
				('Psstar',	Psstar,		False,	None,	None,	None),
				('Tsstar',	Tsstar,		False,	None,	None,	None),
				('Rsstar',	Rsstar,		False,	None,	None,	None),
				('Ms',		Ms,			False,	None,	None,	None),
				('fs',		0.0,		False,	None,	None,	None),
				('verbose',	False,		False,	None,	None,	None))
#fs=0.0 => Pure X0 fit.
#fs=Weight of swelling residual in simultaneous solubility-swelling fit
#Running the Levenberg-Marquart algorithm on the residuals in order to do least squares fitting. This will return the fitted value of the RESIDUALS.
#These need to be added to the experimental datapints to find the fitted pressures.
print('For POLYMER: {} and SOLVENT: {}.'.format(Polymer_Type,Solvent))
print('Using {} parameters.'.format(Parameters_Paper))

fit = minimize(calculateBinaryResidualCondo_Original,params,args=(P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,'X'))

#Reporting the values of the parameters. NEED TO FIGURE OUT HOW TO PRINT THIS TO FILE.
report_fit(fit.params)

