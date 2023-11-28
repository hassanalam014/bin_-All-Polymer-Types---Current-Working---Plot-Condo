# Author: Kier von Konigslow
# Date: November 2017
#
import os,sys,math,csv,matplotlib.pyplot as plt,numpy as npy
# from p_params import *
# from s_params import *
from loadExperimentalData import *
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from loadPhysicalConstants import *
from calculatePureVariables import calculateNewMolecularParameters
# from wrapperFunctions import calculateBinarySolubilitySwelling
# from calculateBinaryResidual import calculateBinarySSQ
from checkResults import *
from sympy import *
import warnings
import cmath
from scipy.optimize import bisect,fsolve
from scipy.interpolate import interp1d
from Parameters_of_Different_Polymers import *
# from All_Functions import *
from Parameters_for_Mixtures_and_Tg import *

###########################################################################
###########################################################################

def SplitExperimental_X_Sw_Data(P0_X_Given,T0_X_Given,X0_X_Given,P0_S_Given,T0_S_Given,S0_S_Given,Rubber0_X_Given,Rubber0_S_Given,Far_Above_Data,**kwargs):

	# print P0_X_Given,T0_X_Given,X0_X_Given,P0_S_Given,T0_S_Given,S0_S_Given,Rubber0_X_Given,Rubber0_S_Given

	for key,value in kwargs.items():
		exec "%s='%s'" % (key,value)

	print 'Polymer Type', Polymer_Type, 'Solvent', Solvent, 'Parameters from', Parameters_Paper, 'Paper_Number', Paper_Number

	#======================================================
	#Post-Thesis Data Below:
	#======================================================
	P0_X = []
	T0_X = []
	X0_X = []
	Rubber0_X = []

	P0_X_below_Tg = []
	T0_X_below_Tg = []
	X0_X_below_Tg = []
	Rubber0_X_below_Tg = []

	P0_X_above_Tg = []
	T0_X_above_Tg = []
	X0_X_above_Tg = []
	Rubber0_X_above_Tg = []

	P0_X_far_above_Tg = []
	T0_X_far_above_Tg = []
	X0_X_far_above_Tg = []
	Rubber0_X_far_above_Tg = []

	P0_S = []
	T0_S = []
	S0_S = []
	Rubber0_S = []

	P0_S_below_Tg = []
	T0_S_below_Tg = []
	S0_S_below_Tg = []
	Rubber0_S_below_Tg = []

	P0_S_above_Tg = []
	T0_S_above_Tg = []
	S0_S_above_Tg = []
	Rubber0_S_above_Tg = []

	P0_S_far_above_Tg = []
	T0_S_far_above_Tg = []
	S0_S_far_above_Tg = []
	Rubber0_S_far_above_Tg = []

	if P0_X_Given!=[]:
		for i in range(0,len(P0_X_Given)):
			if Rubber0_X_Given[i]==0:			
				P0_X_below_Tg += (P0_X_Given[i],)
				T0_X_below_Tg += (T0_X_Given[i],)
				X0_X_below_Tg += (X0_X_Given[i],)
				Rubber0_X_below_Tg += (Rubber0_X_Given[i],)
			elif Rubber0_X_Given[i]==1:			
				P0_X_above_Tg += (P0_X_Given[i],)
				T0_X_above_Tg += (T0_X_Given[i],)
				X0_X_above_Tg += (X0_X_Given[i],)
				Rubber0_X_above_Tg += (Rubber0_X_Given[i],)
			elif Rubber0_X_Given[i]==2:			
				P0_X_far_above_Tg += (P0_X_Given[i],)
				T0_X_far_above_Tg += (T0_X_Given[i],)
				X0_X_far_above_Tg += (X0_X_Given[i],)
				Rubber0_X_far_above_Tg += (Rubber0_X_Given[i],)

		if Far_Above_Data==False:
			P0_X = P0_X_above_Tg
			T0_X = T0_X_above_Tg
			X0_X = X0_X_above_Tg
			Rubber0_X = Rubber0_X_above_Tg

		elif Far_Above_Data==True:
			for i in range(0,len(P0_X_Given)):
				if Rubber0_X_Given[i]==1 or  Rubber0_X_Given[i]==2:			
					P0_X += (P0_X_Given[i],)
					T0_X += (T0_X_Given[i],)
					X0_X += (X0_X_Given[i],)
					Rubber0_X += (Rubber0_X_Given[i],)

	if P0_S_Given!=[]:
		for i in range(0,len(P0_S_Given)):
			if Rubber0_S_Given[i]==0:			
				P0_S_below_Tg += (P0_S_Given[i],)
				T0_S_below_Tg += (T0_S_Given[i],)
				S0_S_below_Tg += (S0_S_Given[i],)
				Rubber0_S_below_Tg += (Rubber0_S_Given[i],)
			elif Rubber0_S_Given[i]==1:			
				P0_S_above_Tg += (P0_S_Given[i],)
				T0_S_above_Tg += (T0_S_Given[i],)
				S0_S_above_Tg += (S0_S_Given[i],)
				Rubber0_S_above_Tg += (Rubber0_S_Given[i],)
			elif Rubber0_S_Given[i]==2:			
				P0_S_far_above_Tg += (P0_S_Given[i],)
				T0_S_far_above_Tg += (T0_S_Given[i],)
				S0_S_far_above_Tg += (S0_S_Given[i],)
				Rubber0_S_far_above_Tg += (Rubber0_S_Given[i],)

		if Far_Above_Data==False:
			P0_S = P0_S_above_Tg
			T0_S = T0_S_above_Tg
			S0_S = S0_S_above_Tg
			Rubber0_S = Rubber0_S_above_Tg

		elif Far_Above_Data==True:

			for i in range(0,len(P0_S_Given)):
				# print 'is it running'
				if Rubber0_S_Given[i]==1 or Rubber0_S_Given[i]==2:			
					P0_S += (P0_S_Given[i],)
					T0_S += (T0_S_Given[i],)
					S0_S += (S0_S_Given[i],)
					Rubber0_S += (Rubber0_S_Given[i],)


	# if P0_X!=[]:
	# 	isotherm_label=1
	# 	number_of_isotherm=1
	# 	isotherm_scanner=T0_X[0]

	# 	for k in range(0,10):
	# 		exec "len_of_isotherm%s=0" %(isotherm_label)
	# 		isotherm_label=isotherm_label+1

	# 	isotherm_label=1

	# 	for k in range(0,len(T0_X)):
	# 		if isotherm_scanner==T0_X[k]:
	# 			exec "len_of_isotherm%s=len_of_isotherm%s+1" %(isotherm_label,isotherm_label)
	# 		elif isotherm_scanner!=T0_X[k]:
	# 			number_of_isotherm = number_of_isotherm+1
	# 			isotherm_scanner=T0_X[k]
	# 			isotherm_label=isotherm_label+1
	# 			exec "len_of_isotherm%s=len_of_isotherm%s+1" %(isotherm_label,isotherm_label)

	# 	print 'number of isotherms in solubility experimental data are:',number_of_isotherm

	# 	isotherm_label=1
	# 	for i in range(0,number_of_isotherm):
	# 		exec "P0_X_T%s=npy.zeros(len_of_isotherm%s)" %(isotherm_label,isotherm_label)
	# 		exec "T0_X_T%s=npy.zeros(len_of_isotherm%s)" %(isotherm_label,isotherm_label)
	# 		exec "X0_X_T%s=npy.zeros(len_of_isotherm%s)" %(isotherm_label,isotherm_label)
	# 		isotherm_label=isotherm_label+1

	# 	isotherm_label=1
	# 	k=0
	# 	for i in range(0,len(T0_X)):
	# 		exec "P0_X_T%s[k]=P0_X[i]" %(isotherm_label)
	# 		exec "T0_X_T%s[k]=T0_X[i]" %(isotherm_label)
	# 		exec "X0_X_T%s[k]=X0_X[i]" %(isotherm_label)
	# 		k=k+1
	# 		if i+2<=len(T0_X):
	# 			if T0_X[i+1]!=T0_X[i]:
	# 				k=0
	# 				isotherm_label=isotherm_label+1

	# 	isotherm_label=1
	# 	for i in range(0,number_of_isotherm):
	# 		exec "print P0_X_T%s" %(isotherm_label)
	# 		exec "print T0_X_T%s" %(isotherm_label)
	# 		exec "print X0_X_T%s" %(isotherm_label)
	# 		isotherm_label=isotherm_label+1

	# if P0_S!=[]:
	# 	isotherm_label=1
	# 	number_of_isotherm=1
	# 	isotherm_scanner=T0_S[0]

	# 	for k in range(0,10):
	# 		exec "len_of_isotherm%s=0" %(isotherm_label)
	# 		isotherm_label=isotherm_label+1

	# 	isotherm_label=1

	# 	for k in range(0,len(T0_S)):
	# 		if isotherm_scanner==T0_S[k]:
	# 			exec "len_of_isotherm%s=len_of_isotherm%s+1" %(isotherm_label,isotherm_label)
	# 		elif isotherm_scanner!=T0_S[k]:
	# 			number_of_isotherm = number_of_isotherm+1
	# 			isotherm_scanner=T0_S[k]
	# 			isotherm_label=isotherm_label+1
	# 			exec "len_of_isotherm%s=len_of_isotherm%s+1" %(isotherm_label,isotherm_label)

	# 	print 'number of isotherms in swelling experimental data are:',number_of_isotherm

	# 	isotherm_label=1
	# 	for i in range(0,number_of_isotherm):
	# 		exec "P0_S_T%s=npy.zeros(len_of_isotherm%s)" %(isotherm_label,isotherm_label)
	# 		exec "T0_S_T%s=npy.zeros(len_of_isotherm%s)" %(isotherm_label,isotherm_label)
	# 		exec "S0_S_T%s=npy.zeros(len_of_isotherm%s)" %(isotherm_label,isotherm_label)
	# 		isotherm_label=isotherm_label+1

	# 	isotherm_label=1
	# 	k=0
	# 	for i in range(0,len(T0_S)):
	# 		exec "P0_S_T%s[k]=P0_S[i]" %(isotherm_label)
	# 		exec "T0_S_T%s[k]=T0_S[i]" %(isotherm_label)
	# 		exec "S0_S_T%s[k]=S0_S[i]" %(isotherm_label)
	# 		k=k+1
	# 		if i+2<=len(T0_S):
	# 			if T0_S[i+1]!=T0_S[i]:
	# 				k=0
	# 				isotherm_label=isotherm_label+1

	# 	isotherm_label=1
	# 	for i in range(0,number_of_isotherm):
	# 		exec "print P0_S_T%s" %(isotherm_label)
	# 		exec "print T0_S_T%s" %(isotherm_label)
	# 		exec "print S0_S_T%s" %(isotherm_label)
	# 		isotherm_label=isotherm_label+1

	return P0_X,P0_X_above_Tg,P0_X_far_above_Tg,T0_X,T0_X_above_Tg,T0_X_far_above_Tg,X0_X,X0_X_above_Tg,X0_X_far_above_Tg,Rubber0_X,Rubber0_X_above_Tg,Rubber0_X_far_above_Tg,P0_S,P0_S_above_Tg,P0_S_far_above_Tg,T0_S,T0_S_above_Tg,T0_S_far_above_Tg,S0_S,S0_S_above_Tg,S0_S_far_above_Tg,Rubber0_S,Rubber0_S_above_Tg,Rubber0_S_far_above_Tg