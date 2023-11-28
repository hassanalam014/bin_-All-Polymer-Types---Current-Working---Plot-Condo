# Author: Kier von Konigslow
# Date: May 2017
#
# Description	: The purpose of this file is to estimate the multicomponent fluid parameters
#				  for the PS/CO2 binary mixture.
#

import os,sys,math,csv,numpy as npy
from isListOrNpyArray import *
import warnings
# from p_params import *
# from s_params import *
from loadExperimentalData import *
from lmfit import minimize, Parameters, report_fit
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
# from calculateBinaryResidual import calculateBinaryResidualCHV
from All_Functions import *
# from calculateBinaryVariablesCHV import *
from Parameters_of_Different_Polymers import *
from SplittingExperimentalDataOf_X_Sw_aboveANDbelowTgANDIsotherms import *
from Tait_Parameters_of_Different_Polymers import *
from loadExperimentalDataCO2 import *
from CO2PVT_interpolation import *
from Split_Exp_Data_in_Isotherms import*

def residualFunction(A0,A,weight=1.0):
	if len(A0) != len(A):
		raise ValueError('In residual: The number of experimental points and number of theoretical points are not equal.')
	
	residual = npy.zeros(len(A0))
	
	for i in range(0,len(A0)):
		residual[i] = weight*((A0[i]-A[i]))/A0[i]  #Kier original had no hash and no absolute
	
	
	# print 'weight is', weight
	return residual

def binaryResidual(theory,Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,**kwargs):
	print fs
	m_s = npy.zeros(len(P0_X))
	Sw = npy.zeros(len(P0_S))
	res_X = npy.zeros(len(P0_X))
	res_S = npy.zeros(len(P0_S))
	
	if len(P0_X) != len(P0_S):
		warnings.warn('In binaryResidual: Mismatch in solubility and swelling data number. Results may be skewed.')

	# suppress_print = kwargs.pop('suppress_print',False)
	# if not suppress_print:

	# for key,value in kwargs.items():
	# 	print '%s=%s' % (key,value)

	# for key,value in kwargs.items():
	# 	exec "%s=%s" % (key,value)
	
	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))

	if 'X' in fit_type:

		result = calculateBinarySolubilitySwelling(theory,P0_X,T0_X,Mp,Ms,**kwargs)
		m_s = result[2]
		Sw = result[3]
		phip = result[4]
		phis = result[5]
		# answer= calculateCorrectSolubilityExp(P0_X,T0_X,X0_X,Rgas,vppure,m_s,Sw,phip,phis,Mp,Ms,**kwargs)
		# m_s_corrected = answer[0]
		res_X = residualFunction(X0_X,m_s,1.0-fs)
	if 'S' in fit_type:
		result = calculateBinarySolubilitySwelling(theory,P0_S,T0_S,Mp,Ms,**kwargs)
		Sw = result[3]
		# P0,T0,Sw = calculateBinarySwelling(theory,P0_S,T0_S,Mp,Ms,**kwargs)
		# print 'swelling weight is'
		res_S = residualFunction(S0_S,Sw,fs)
	
	if 'X' in fit_type and 'S' in fit_type:
		residual = npy.concatenate((res_X,res_S),axis=0)
	elif 'X' in fit_type:
		residual = res_X
	elif 'S' in fit_type:
		residual = res_S
	else:
		raise ValueError('In binaryResidual: fit_type must contain X and/or S.')

	return residual

def calculateBinaryResidualCHV(params,Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,fit_type,method):

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
	
	if 'k12' in params and 'delta' in params:
		k12 = params['k12'].value
		delta = params['delta'].value
		kwargs.update({'k12':k12,'delta':delta})
	elif 'zeta' in params and 'delta' in params:
		zeta = params['zeta'].value
		delta = params['delta'].value
		kwargs.update({'zeta':zeta,'delta':delta})
	else:
		raise ValueError('In calculateBinaryResidualCHV, mixture parameters: (k12,delta) or (zeta,delta) mixture parameters must be used.')
	
	if method == 'disparate':
		kwargs.update({'method':'disparate'})
	elif method == 'single':
		kwargs.update({'method':'single'})
	elif method == 'mixed':
		kwargs.update({'method':'mixed'})
	else:
		warnings.warn("In calculateBinaryResidualCHV, method: method parameter not specified. method = 'disparate' will be used.")
	
	if 'verbose' in params:
		verbose = params['verbose'].value
		kwargs.update({'verbose':verbose})

	if 'Kier' in params and 'Hassan' in params and 'Condo' in params:
		Kier = params['Kier'].value
		Hassan = params['Hassan'].value
		Condo = params['Condo'].value
		Hassan_Var_Vol = params['Hassan_Var_Vol'].value
		kwargs.update({'Kier':Kier,'Hassan':Hassan,'Condo':Condo,'Hassan_Var_Vol':Hassan_Var_Vol})

	if 'forward' in params and 'backward' in params:
		forward = params['forward'].value
		backward = params['backward'].value
		kwargs.update({'forward':forward,'backward':backward})
	
	res = binaryResidual('CHV',Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,**kwargs)
	
	print '==> Done for values above.'

	return res

Polymer_Type='PMMA'
Solvent='CO2'

Parameters_Paper ='Self_Grassia'			# P*T*R* and g,epsilon_2,x (PVT-Tg Data Paper or Direct P*T*R* Values Reference)
Cp_Polymer_Weight = '02kilo_POST_THESIS'	# g,epsilon_2,x (Cp Paper Reference)
Paper_Number = 'Paper15'						# Solubility or Swelling Data Reference

kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight}

Ppstar,Tpstar,Rpstar,Mp,Psstar,Tsstar,Rsstar,Ms,P_exp,Tg_exp=Parameters_of_Different_Polymers(**kwargs)
P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete=loadExperimentSwXData(**kwargs)
Far_Above_Data=False
P0_X,P0_X_above_Tg,P0_X_far_above_Tg,T0_X,T0_X_above_Tg,T0_X_far_above_Tg,X0_X,X0_X_above_Tg,X0_X_far_above_Tg,Rubber0_X,Rubber0_X_above_Tg,Rubber0_X_far_above_Tg,P0_S,P0_S_above_Tg,P0_S_far_above_Tg,T0_S,T0_S_above_Tg,T0_S_far_above_Tg,S0_S,S0_S_above_Tg,S0_S_far_above_Tg,Rubber0_S,Rubber0_S_above_Tg,Rubber0_S_far_above_Tg = SplitExperimental_X_Sw_Data(P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete,Far_Above_Data,**kwargs)
v_0,alpha,B0,B1 = Tait_Parameters_of_Different_Polymers(**kwargs)

# P0_X = [6.89, 20.68]
# T0_X = [453.0, 453.0] 
# X0_X = [0.040122864, 0.087008126]

result = calculateR_interCO2(P0_X,T0_X,P0_isobars,T0_isobars,R0_isobars)
Rgas = npy.array(result[0])
result = calculateTaitVolume(P0_X,T0_X,v_0,alpha,B0,B1)
vppure = npy.array(result[0])

number_of_isotherm, result = Split_Isotherms(P0_X,T0_X,X0_X,'X')
P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5,P0_X_T6,T0_X_T6,X0_X_T6,P0_X_T7,T0_X_T7,X0_X_T7,P0_X_T8,T0_X_T8,X0_X_T8 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23]
# print P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5

number_of_isotherm_swelling, result = Split_Isotherms(P0_S,T0_S,S0_S,'S')
P0_S_T1,T0_S_T1,S0_S_T1,P0_S_T2,T0_S_T2,S0_S_T2,P0_S_T3,T0_S_T3,S0_S_T3,P0_S_T4,T0_S_T4,S0_S_T4,P0_S_T5,T0_S_T5,S0_S_T5,P0_S_T6,T0_S_T6,S0_S_T6,P0_S_T7,T0_S_T7,S0_S_T7,P0_S_T8,T0_S_T8,S0_S_T8 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23]
# print P0_S_T1,T0_S_T1,S0_S_T1,P0_S_T2,T0_S_T2,S0_S_T2,P0_S_T3,T0_S_T3,S0_S_T3,P0_S_T4,T0_S_T4,S0_S_T4,P0_S_T5,T0_S_T5,S0_S_T5

P0_X = P0_X_T2
T0_X = T0_X_T2
X0_X = X0_X_T2

P0_S = P0_S_T2
T0_S = T0_S_T2
S0_S = S0_S_T2

# zeta:            0.99708224 +/- 0.00280651 (0.28%) (init = 1.02)
# delta:           0.95000002 +/- 0.00929252 (0.98%) (init = 0.97)

# zeta:            0.97583647 +/- 0.00262431 (0.27%) (init = 1.02)
# delta:           0.92000000 +/- 0.00509818 (0.55%) (init = 0.97)
Kier=False
Hassan=True  
Condo=False  
Hassan_Var_Vol=False

forward=False
backward=True 
# 
#Initializing the parameters.
params = Parameters()
#The following code sets up the model's parameters. It includes both fitting parameters and parameters that will remain fixed
#for the fitting. The values given are the inital guesses of fitting parameters and values of fixed parameters.
#           	(Name,			Value,		Vary?,	Min,	Max,	Expr)
params.add_many(('zeta',		0.97583,		True,	0.92,	1.5,	None),
				('delta',		0.92,		True,	0.80,	1.2,	None),
				('Ppstar',		Ppstar,		False,	None,	None,	None),
				('Tpstar',		Tpstar,		False,	None,	None,	None),
				('Rpstar',		Rpstar,		False,	None,	None,	None),
				('Mp',			Mp,			False,	None,	None,	None),
				('Psstar',		Psstar,		False,	None,	None,	None),
				('Tsstar',		Tsstar,		False,	None,	None,	None),
				('Rsstar',		Rsstar,		False,	None,	None,	None),
				('Ms',			Ms,			False,	None,	None,	None),
				('Kier',		Kier,		False,	None,	None,	None),
				('Hassan',		Hassan,		False,	None,	None,	None),
				('Condo',		Condo,		False,	None,	None,	None),
				('Hassan_Var_Vol',	Hassan_Var_Vol,		False,	None,	None,	None),
				('forward',		forward,	False,	None,	None,	None),
				('backward',	backward,	False,	None,	None,	None),
				('fs',			0.0,		False,	None,	None,	None),
				('verbose',		False,		False,	None,	None,	None))

#fs=Weight of swelling residual in simultaneous solubility-swelling fit
#Running the Levenberg-Marquart algorithm on the residuals in order to do least squares fitting. This will return the fitted value of the RESIDUALS.
#These need to be added to the experimental datapints to find the fitted pressures.
print('For POLYMER: {} and SOLVENT: {}.'.format(Polymer_Type,Solvent))
print('Using {} parameters.'.format(Parameters_Paper))

#ftol=dependent variable toleracne, xtol=independent variable tolerance, gtol=gradiant tolerance. Default are 1e-08.

fit = minimize(calculateBinaryResidualCHV,params,args=(Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,'X','disparate'))
#																						#disparate is dummy variable.

#Reporting the values of the parameters. NEED TO FIGURE OUT HOW TO PRINT THIS TO FILE.
report_fit(fit.params)