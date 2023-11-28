# # Author: Kier von Konigslow
# # Date: November 2017
# #
import os,sys,math,csv,matplotlib.pyplot as plt,numpy as npy
# # from p_params import *
# # from s_params import *
# from loadExperimentalData import *
# lib_path = os.path.abspath(os.path.join('..'))
# sys.path.append(lib_path)
# from loadPhysicalConstants import *
# from calculatePureVariables import calculateNewMolecularParameters
# # from wrapperFunctions import calculateBinarySolubilitySwelling
# # from calculateBinaryResidual import calculateBinarySSQ
# from checkResults import *
# from sympy import *
# import warnings
# import cmath
# from scipy.optimize import bisect,fsolve
# from scipy.interpolate import interp1d
# from Parameters_of_Different_Polymers import *
# from All_Functions import *
# # from All_Functions_brentsolve import *
# from Parameters_for_Mixtures_and_Tg import *
# from SplittingExperimentalDataOf_X_Sw_aboveANDbelowTgANDIsotherms import *
# from Tait_Parameters_of_Different_Polymers import *
# from loadExperimentalDataCO2 import *
# from CO2PVT_interpolation import *
# ###########################################################################
# ###########################################################################

# Polymer_Type='PMMA'
# Solvent='CO2'
# Parameters_Paper ='Self_Grassia'			# P*T*R* and g,epsilon_2,x (PVT-Tg Data Paper or Direct P*T*R* Values Reference)
# Cp_Polymer_Weight = '02kilo_POST_THESIS'	# g,epsilon_2,x (Cp Paper Reference)
# Paper_Number = 'Paper9'						# Solubility or Swelling Data Reference

# kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight}

# Ppstar,Tpstar,Rpstar,Mp,Psstar,Tsstar,Rsstar,Ms,P_exp,Tg_exp=Parameters_of_Different_Polymers(**kwargs)
# P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete=loadExperimentSwXData(**kwargs)
# Far_Above_Data=True
# P0_X,P0_X_above_Tg,P0_X_far_above_Tg,T0_X,T0_X_above_Tg,T0_X_far_above_Tg,X0_X,X0_X_above_Tg,X0_X_far_above_Tg,Rubber0_X,Rubber0_X_above_Tg,Rubber0_X_far_above_Tg,P0_S,P0_S_above_Tg,P0_S_far_above_Tg,T0_S,T0_S_above_Tg,T0_S_far_above_Tg,S0_S,S0_S_above_Tg,S0_S_far_above_Tg,Rubber0_S,Rubber0_S_above_Tg,Rubber0_S_far_above_Tg = SplitExperimental_X_Sw_Data(P0_X_complete,T0_X_complete,X0_X_complete,P0_S_complete,T0_S_complete,S0_S_complete,Rubber0_X_complete,Rubber0_S_complete,Far_Above_Data,**kwargs)
# v_0,alpha,B0,B1 = Tait_Parameters_of_Different_Polymers(**kwargs)

def Split_Isotherms(P0,T0,XS0,property):
	
	number_of_isotherm_expected = 20
	isotherm_expected = 1
	for k in range(0,number_of_isotherm_expected):
		exec "P0_%s_T%s=[]" %(property,isotherm_expected)
		exec "T0_%s_T%s=[]" %(property,isotherm_expected)
		exec "%s0_%s_T%s=[]" %(property,property,isotherm_expected)
		isotherm_expected=isotherm_expected+1

	if T0!=[]:
		isotherm_label=1
		number_of_isotherm=1
		isotherm_scanner=T0[0]

		for k in range(0,10):
			exec "len_of_isotherm%s=0" %(isotherm_label)
			isotherm_label=isotherm_label+1

		isotherm_label=1

		for k in range(0,len(T0)):
			if isotherm_scanner==T0[k]:
				exec "len_of_isotherm%s=len_of_isotherm%s+1" %(isotherm_label,isotherm_label)
			elif isotherm_scanner!=T0[k]:
				number_of_isotherm = number_of_isotherm+1
				isotherm_scanner=T0[k]
				isotherm_label=isotherm_label+1
				exec "len_of_isotherm%s=len_of_isotherm%s+1" %(isotherm_label,isotherm_label)

		print 'number of isotherms are:',number_of_isotherm

		isotherm_label=1
		for i in range(0,number_of_isotherm):
			exec "P0_%s_T%s=npy.zeros(len_of_isotherm%s)" %(property,isotherm_label,isotherm_label)
			exec "T0_%s_T%s=npy.zeros(len_of_isotherm%s)" %(property,isotherm_label,isotherm_label)
			exec "%s0_%s_T%s=npy.zeros(len_of_isotherm%s)" %(property,property,isotherm_label,isotherm_label)
			isotherm_label=isotherm_label+1

		isotherm_label=1
		k=0
		for i in range(0,len(T0)):
			exec "P0_%s_T%s[k]=P0[i]" %(property,isotherm_label)
			exec "T0_%s_T%s[k]=T0[i]" %(property,isotherm_label)
			exec "%s0_%s_T%s[k]=XS0[i]" %(property,property,isotherm_label)
			k=k+1
			if i+2<=len(T0):
				if T0[i+1]!=T0[i]:
					k=0
					isotherm_label=isotherm_label+1

		isotherm_label=1
		for i in range(0,number_of_isotherm):
			exec "print 'P0_%s_T%s=',P0_%s_T%s" %(property,isotherm_label,property,isotherm_label)
			exec "print 'T0_%s_T%s=',T0_%s_T%s" %(property,isotherm_label,property,isotherm_label)
			exec "print '%s0_%s_T%s=',%s0_%s_T%s" %(property,property,isotherm_label,property,property,isotherm_label)
			isotherm_label=isotherm_label+1

	result = [] 
	isotherm_label=1
	for i in range(0,number_of_isotherm_expected):
		exec "result.append(P0_%s_T%s)" %(property,isotherm_label)
		exec "result.append(T0_%s_T%s)" %(property,isotherm_label)
		exec "result.append(%s0_%s_T%s)" %(property,property,isotherm_label)
		isotherm_label=isotherm_label+1	       

	return number_of_isotherm, result







'''
Flatoooooooooooooooooo!

if T0!=[]:
	isotherm_label=1
	number_of_isotherm=1
	isotherm_scanner=T0[0]

	for k in range(0,10):
		exec "len_of_isotherm%s=0" %(isotherm_label)
		isotherm_label=isotherm_label+1

	isotherm_label=1

	for k in range(0,len(T0)):
		if isotherm_scanner==T0[k]:
			exec "len_of_isotherm%s=len_of_isotherm%s+1" %(isotherm_label,isotherm_label)
		elif isotherm_scanner!=T0[k]:
			number_of_isotherm = number_of_isotherm+1
			isotherm_scanner=T0[k]
			isotherm_label=isotherm_label+1
			exec "len_of_isotherm%s=len_of_isotherm%s+1" %(isotherm_label,isotherm_label)


	print 'number of isotherms are:',number_of_isotherm

	isotherm_label=1
	for i in range(0,number_of_isotherm):
		exec "P0_T%s=npy.zeros(len_of_isotherm%s)" %(isotherm_label,isotherm_label)
		exec "T0_T%s=npy.zeros(len_of_isotherm%s)" %(isotherm_label,isotherm_label)
		exec "XS0_T%s=npy.zeros(len_of_isotherm%s)" %(isotherm_label,isotherm_label)
		isotherm_label=isotherm_label+1

	isotherm_label=1
	k=0
	for i in range(0,len(T0)):
		exec "P0_T%s[k]=P0[i]" %(isotherm_label)
		exec "T0_T%s[k]=T0[i]" %(isotherm_label)
		exec "XS0_T%s[k]=XS0[i]" %(isotherm_label)
		k=k+1
		if i+2<=len(T0):
			if T0[i+1]!=T0[i]:
				k=0
				isotherm_label=isotherm_label+1

	isotherm_label=1
	for i in range(0,number_of_isotherm):
		exec "print 'P0_T%s=',P0_T%s" %(isotherm_label,isotherm_label)
		exec "print 'T0_T%s=',T0_T%s" %(isotherm_label,isotherm_label)
		exec "print 'XS0_T%s=',XS0_T%s" %(isotherm_label,isotherm_label)
		isotherm_label=isotherm_label+1

'''
