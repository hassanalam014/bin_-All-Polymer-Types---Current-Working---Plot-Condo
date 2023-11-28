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
from calculateBinaryResidual import calculateBinaryResidualCHV
from Parameters_of_Different_Polymers import *
from SplittingExperimentalDataOf_X_Sw_aboveANDbelowTgANDIsotherms import *
from Parameters_of_Different_Polymers import *
from Split_Exp_Data_in_Isotherms import*

Polymer_Type='PMMA'
Solvent='CO2'

Parameters_Paper ='Self_Grassia'			# P*T*R* and g,epsilon_2,x (PVT-Tg Data Paper or Direct P*T*R* Values Reference)
Cp_Polymer_Weight = '02kilo_POST_THESIS'	# g,epsilon_2,x (Cp Paper Reference)
Paper_Number = 'Paper15'						# Solubility or Swelling Data Reference
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

vs = kB*Tsstar/Psstar
vp = kB*Tpstar/Ppstar
delta_min = vs/vs
delta_max = vp/vs
print delta_min
print delta_max

#Initializing the parameters.
params = Parameters()
#The following code sets up the model's parameters. It includes both fitting parameters and parameters that will remain fixed
#for the fitting. The values given are the inital guesses of fitting parameters and values of fixed parameters.
#           	(Name,		Value,		Vary?,	Min,	Max,	Expr)
params.add_many(('zeta',	1.10,		True,	None,	None,	None),
				('delta',	1.2,	False,	delta_min,	delta_max,	None),
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

fit = minimize(calculateBinaryResidualCHV,params,args=(P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,'X','disparate'))

#Reporting the values of the parameters. NEED TO FIGURE OUT HOW TO PRINT THIS TO FILE.
report_fit(fit.params)