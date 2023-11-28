# Author: Kier von Konigslow
# Date: May 2017
#
# Description: The purpose of this file is to plot the saturated PMMA/CO2 binary mixture information
#			   based on both experimental and theoretical data.
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
from calculateBinaryResidual import calculateBinarySSQ
from calculateBinaryVariablesCHV import *
from Parameters_of_Different_Polymers import *
from Parameters_for_Mixtures_and_Tg import *
from All_Functions import calculateThermodynamicVariables
from SplittingExperimentalDataOf_X_Sw_aboveANDbelowTgANDIsotherms import *
# from Tait_Parameters_of_Different_Polymers import *
# from loadExperimentalDataCO2 import *
# from CO2PVT_interpolation import *
from Split_Exp_Data_in_Isotherms import*
from collections import OrderedDict			#For Exotic Line Styles
import cmath
from sympy import *
import types

def remove_duplicates(lst):
    res = []
    for x in lst:
        if x not in res:
            res.append(x)
    return res

def binaryPhaseEquilibriumCHV(P,T,Mp,Ms,**kwargs):

	#Reference:
	# -p --> polymer
	# -s --> solvent

	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))
	
	if 'alpha_p' in kwargs and 'vhp' in kwargs and 'epsilon_p' in kwargs:
		Ppstar,Tpstar,Vpstar = calculateCharacteristicParameters(alpha_p,vhp,epsilon_p,Mp)
	elif 'Ppstar' in kwargs and 'Tpstar' in kwargs and 'Rpstar' in kwargs:
		pass
	else:
		raise ValueError('In binaryPhaseEquilibriumCHV, polymer parameters: Either molecular (alpha_p,vhp,epsilon_p) or characteristic (Ppstar,Tpstar,Rpstar) parameters must be passed into keyword arguments.')
	
	if 'alpha_s' in kwargs and 'vhs' in kwargs and 'epsilon_s' in kwargs:
		Psstar,Tsstar,Vsstar = calculateCharacteristicParameters(alpha_s,vhs,epsilon_s,Ms)
	elif 'Psstar' in kwargs and 'Tsstar' in kwargs and 'Rsstar' in kwargs:
		pass
	else:
		raise ValueError('In binaryPhaseEquilibriumCHV, solvent parameters: Either molecular (alpha_s,vhs,epsilon_s) or characteristic (Psstar,Tsstar,Rsstar) parameters must be passed into keyword arguments.')
	
	if 'k12' in kwargs and 'delta' in kwargs:
		pass
	elif 'zeta' in kwargs and 'delta' in kwargs:
		pass
	else:
		raise ValueError('In binaryPhaseEquilibriumCHV, mixture parameters: (k12,delta) or (zeta,delta) mixture parameters must be passed into keyword arguments.')
	
	#Allows for method argument in kwargs. Options are: 'disparate', 'single', 'mixed'.
	#Default option is 'disparate'.
	# -'disparate'	--> Mixture phase has constant hole volume of vhm. Pure phases have constant hole volumes vhp, vhs.
	# -'single'		--> All phases have hole volume vhm.
	method = kwargs.pop('method','disparate')
	
	#Boolean determining whether information is printed.
	#Default option is False.
	verbose = kwargs.pop('verbose',False)
	if verbose:
		print('FOR: P = {}MPa, T = {}K;'.format(P,T))
	
	#Initializing phi_p, phi_s and v_h as symbolic variables for the sympy package.
	#This is a step necessary for the numerical solver nsolve.
	phi_p = Symbol('phi_p',real=True)
	phi_s = Symbol('phi_s',real=True)
	v_h = Symbol('v_h',real=True)
	
	#PURE FLUID PARAMETERS.
	vpp = Mp/Rpstar				#Hassan: This is volume of whole polymer chain N_k*v_k. Not only segment of chain v_k.
	vss = Ms/Rsstar
	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar

	# vrp = kB*Tpstar/Ppstar
	# vrs = kB*Tsstar/Psstar
	# print 'vhp is:', vhp
	# print 'vhs is:', vhs

	chi_pp = -2*Tpstar/T			#Hassan: He has introduced a new variable not mentioned in paper.
	chi_ss = -2*Tsstar/T			#Hassan: He has introduced a new variable not mentioned in paper.
	alpha_p0 = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_s0 = (Psstar*Ms)/(kB*Tsstar*Rsstar)

	# print 'alpha_s0 is:', alpha_s0
	# print 'alpha_p0 is:', alpha_p0
	
	# vhs=vrs
	# Pfalse= 122.838243
	# vhp=kB*Tpstar/Pfalse

	#MIXTURE PARAMETERS.
	if 'k12' in kwargs:
		chi_ps = -(1.0-k12)*math.sqrt(chi_pp*chi_ss)
	elif 'zeta' in kwargs:
		chi_ps = -zeta*math.sqrt(chi_pp*chi_ss)				#Hassan: This is Kier paper Eq(34)
	vhm = delta*vhs											#Hassan: This is definition of delta
	alpha_p = alpha_p0*vhp/v_h								#Hassan: v_h is volume of hole in mixture or pure solvent
	alpha_s = alpha_s0*vhs/v_h
	
	# alpha_0 = v_h/vhs			#Hassan: Self Added Line
	# alpha_p0 = alpha_p0*vhp/vhs #Hassan: Self Added Line
	# alpha_p = alpha_p0			#Hassan: Self Added Line
	# alpha_s = alpha_s0			#Hassan: Self Added Line

	# vhs=alpha_0*vrs
	# print 'vhs is:', vhs
	# print 'vrs is:', vrs
	# vhp=alpha_0*vrp


	#Hong and Nulandi limiting value for ln_vh. THIS SHOULD BE RECONSIDERED.
	ln_vh = 1.0
	# print 'am i running'
	#EQUATION OF STATE, CHEMICAL POTENTIAL.
	#Mixture equation of state.
	EOS = v_h*P/(kB*T)-(chi_pp/2)*phi_p**2-chi_ps*phi_p*phi_s-(chi_ss/2)*phi_s**2+(1.0-1.0/alpha_p)*phi_p+(1.0-1.0/alpha_s)*phi_s+log(1.0-phi_p-phi_s)
	# EOS = v_h*P/(kB*T)-(chi_pp/2)*phi_p**2-chi_ps*phi_p*phi_s-(chi_ss/2)*phi_s**2+phi_p+(1.0-1.0/alpha_s)*phi_s+log(1.0-phi_p-phi_s)
	# EOS = vhs*P/(kB*T)-(chi_pp/2)*phi_p**2-chi_ps*phi_p*phi_s-(chi_ss/2)*phi_s**2+((1.0/alpha_0)-(1.0/alpha_p))*phi_p+((1.0/alpha_0)-(1.0/alpha_s))*phi_s+(1/alpha_0)*log(1.0-phi_p-phi_s)  #Hassan: My Modification of reference volume

	#Mixture equation of state in general.
	EOS_m = EOS.subs([(phi_p,phi_p),(phi_s,phi_s),(v_h,vhm)])
	
	#Mixture solvent chemical potential.
	# mu_s = chi_ss*phi_s+chi_ps*phi_p+(1.0/alpha_s)*(0+log(phi_s))-log(1-phi_p-phi_s)-ln_vh  #This seems wrong.
	mu_s = alpha_s*(chi_ss*phi_s+chi_ps*phi_p+(1.0/alpha_s)*(1+log(phi_s))-log(1-phi_p-phi_s)-1)  #Hassan: This is my correction.
	# mu_s = (chi_ss*phi_s+chi_ps*phi_p+(1.0/alpha_s)*(1+log(phi_s))-log(1-phi_p-phi_s)-1)  #Hassan: Prefactor alpha_k discarded
	# mu_s = (chi_ss*phi_s+chi_ps*phi_p+(1.0/alpha_s)*(1+log(phi_s))-(1/alpha_0)*log(1-phi_p-phi_s)-(1/alpha_0))  #Hassan: This is my correction when reference volume is not equal hole volume

	#Mixture solvent chemical potential in general.
	mu_s_m = mu_s.subs([(phi_p,phi_p),(phi_s,phi_s),(v_h,vhm)])

	#Mixture polymer chemical potential.
	# mu_p = chi_pp*phi_p+chi_ps*phi_s+(1.0/alpha_p)*(0+log(phi_p))-log(1-phi_p-phi_s)-ln_vh #This seems wrong.
	mu_p = alpha_p*(chi_pp*phi_p+chi_ps*phi_s+(1.0/alpha_p)*(1+log(phi_p))-log(1-phi_p-phi_s)-1) #Hassan: This is my correction
	# mu_p = (chi_pp*phi_p+chi_ps*phi_s+(1.0/alpha_p)*(1+log(phi_p))-log(1-phi_p-phi_s)-1) #Hassan: Prefactor alpha_k discarded
	# mu_p = (chi_pp*phi_p+chi_ps*phi_s+(1.0/alpha_p)*(1+log(phi_p))-(1/alpha_0)*log(1-phi_p-phi_s)-(1/alpha_0)) #Hassan: This is my correction when reference volume is not equal hole volume

	#Mixture polymer chemical potential in general.
	mu_p_m = mu_p.subs([(phi_p,phi_p),(phi_s,phi_s),(v_h,vhm)])
	
	#Selecting appropriate approach to use for limiting hole volumes.
	#Refers to 'method' in kwargs.
	if method == 'disparate':
		#Mixture equation of state in phi_s --> 0 limit. v_h takes on value vhp.
		EOS_p0 = EOS.subs([(phi_p,phi_p),(phi_s,0.0),(v_h,vhp)])
		#Mixture equation of state in phi_p --> 0 limit. v_h takes on value vhs.
		EOS_s0 = EOS.subs([(phi_p,0.0),(phi_s,phi_s),(v_h,vhs)])
		#Mixture solvent chemical potential in phi_p --> 0 limit. v_h takes on value vhs.
		mu_s0 = mu_s.subs([(phi_p,0.0),(phi_s,phi_s),(v_h,vhs)])
	if method == 'single':
		# print 'is this running'
		#Mixture equation of state in phi_s --> 0 limit. v_h takes on value vhm.
		EOS_p0 = EOS.subs([(phi_p,phi_p),(phi_s,0.0),(v_h,vhm)])
		#Mixture equation of state in phi_p --> 0 limit. v_h takes on value vhm.
		EOS_s0 = EOS.subs([(phi_p,0.0),(phi_s,phi_s),(v_h,vhm)])
		#Mixture solvent chemical potential in phi_p --> 0 limit. v_h takes on value vhm.
		mu_s0 = mu_s.subs([(phi_p,0.0),(phi_s,phi_s),(v_h,vhm)])
	if method == 'mixed':
		#Mixture equation of state in phi_s --> 0 limit. v_h takes on value vhp.
		EOS_p0 = EOS.subs([(phi_p,phi_p),(phi_s,0.0),(v_h,vhp)])
		#Mixture equation of state in phi_p --> 0 limit. v_h takes on value vhs.
		EOS_s0 = EOS.subs([(phi_p,0.0),(phi_s,phi_s),(v_h,vhs)])
		#Mixture solvent chemical potential in phi_p --> 0 limit. v_h takes on value vhm.
		mu_s0 = mu_s.subs([(phi_p,0.0),(phi_s,phi_s),(v_h,vhm)])

	print('P = {}, T = {};'.format(P,T))

	#CALCULATION OF PURE FLUID STATE AT P, T.
	#Solving for the volume fraction of the system in the pure fluid limiting cases.
	
	phip0 = 0.0
	phip0_all_values = []

	guess1 = npy.array([0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99,0.999,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.01,0.02,0.05,0.9999,0.99999,0.999999,0.000001,0.00001,0.0001,0.001])
	# guess1 = npy.array([0.50,0.65,0.75,0.85,0.90,0.97,0.99,0.999,0.10,0.20,0.30,0.40,0.01,0.02,0.05,0.9999,0.0001,0.001])

	for i in range(len(guess1)):
		try:
			phip0 = nsolve(EOS_p0,phi_p,guess1[i],verify=True)
		except:
			pass
		
		phip0 = complex(phip0)

		if phip0.real>0.0 and abs(phip0.imag)<=10E-3:
			# print 'Is phip0 complex:',phip0
			phip0 = abs(phip0)
			phip0 = round(phip0, 6)
			print 'Hurry! phip0 is:', phip0
			phip0_all_values.append(phip0)
			# break
		else:
			phip0 = 0.0

	if phip0==0.0:
		print 'Program Failed to get value of phip0'
		# raise ValueError('Program Failed to get value of phip0')

	phip0_all_values = npy.array(remove_duplicates(phip0_all_values))
	print phip0_all_values
	phip0 = phip0_all_values[0]

	for i in range(len(phip0_all_values)-1):
		if phip0_all_values[i+1]>phip0_all_values[i]:
			phip0 = phip0_all_values[i+1]
		else:
			phip0 = phip0_all_values[i]		

	phis0 = 0.0
	phis0_all_values = []

	guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.000001,0.00001,0.90,0.95,0.97,0.98,0.99,0.999,0.9999,0.99999,0.999999])
	# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.65,0.75,0.85,0.00001,0.90,0.97,0.99,0.999,0.9999])
	for i in range(len(guess2)):
		try:
			phis0 = nsolve(EOS_s0,phi_s,guess2[i],verify=True)
		except:
			pass

		phis0 = complex(phis0)

		if phis0.real>0.0 and abs(phis0.imag)<=10E-3:
			# print 'Is phis0 complex:',phis0
			phis0 = abs(phis0)
			phis0 = round(phis0, 6)
			print 'Hurry! phis0 is:', phis0
			phis0_all_values.append(phis0)
			# break
		else:
			phis0 = 0.0

	if phis0==0.0:
		print 'Program Failed to get value of phis0'
		# raise ValueError('Program Failed to get value of phis0')

	phis0_all_values = npy.array(remove_duplicates(phis0_all_values))
	print phis0_all_values
	phis0 = phis0_all_values[0]

	for i in range(len(phis0_all_values)-1):
		if phis0_all_values[i+1]>phis0_all_values[i]:
			phis0 = phis0_all_values[i+1]
		else:
			phis0 = phis0_all_values[i]		

	print 'Is phip0 complex:',phip0
	print 'Is phis0 complex:',phis0

	# phip0=abs(phip0)
	# phis0=abs(phis0)

	#CHECKING IF PURE VOLUME FRACTION RESULTS ARE VALID.
	checkVolumeFraction(phip0,'phi_p')
	checkVolumeFraction(phis0,'phi_s')
	
	#PRINTING OF RESULTS OF PURE FLUID CALCULATIONS.
	#FOR DIAGNOSTIC PURPOSES.
	if verbose:
		print('phip0 = {}, phis0 = {};'.format(phip0,phis0))
	
	# mu_gasPhase=mu_s0.subs(phi_s,phis0)
	# print mu_gasPhase

	#CALCULATION OF BINARY MIXTURE COMPOSITION AT P, T.
	#default [0.75,0.05]
	#Other good range [0.85,0.10]

	phip = 0.0
	phis = 0.0

	phip_all_values = []
	phis_all_values = []

	# guess_phip = npy.array([0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99,0.999,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.01,0.02,0.05,0.9999,0.99999,0.999999,0.000001,0.00001,0.0001,0.001])
	# guess_phis = npy.array([0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.0001,0.001,0.000001,0.00001,0.90,0.95,0.97,0.98,0.99,0.999,0.9999,0.99999,0.999999])
	guess_phip = npy.array([0.50,0.65,0.75,0.85,0.90,0.97,0.99,0.999,0.10,0.20,0.30,0.40,0.01,0.02,0.05,0.9999,0.0001,0.001])
	guess_phis = npy.array([0.001,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.65,0.75,0.85,0.0001,0.00001,0.90,0.97,0.99,0.999,0.9999])

	for i in range(len(guess_phip)):
		for j in range(len(guess_phis)):
			try:
				phip,phis = nsolve([EOS_m,(mu_s_m-mu_s0.subs(phi_s,phis0))],[phi_p,phi_s],[guess_phip[i],guess_phis[j]],verify=True)
			except:
				pass

			phip = complex(phip)
			phis = complex(phis)

			if phip.real>0.0 and abs(phip.imag)<=10E-3 and phis.real>0.0 and abs(phis.imag)<=10E-15:
				# print 'Is phip complex:',phip
				# print 'Is phis complex:',phis
				phip = abs(phip)
				phip = round(phip, 6)
				phis = abs(phis)
				phis = round(phis, 6)
				print 'Hurry! phip is:', phip, 'and phis is:', phis
				phip_all_values.append(phip)
				phis_all_values.append(phis)				
				break
			else:
				phip = 0.0
				phis = 0.0

			# if phip>0.0 and phis>0.0:
			# 	print 'Hurry! phip is:', phip, 'and phis is:', phis
			# 	break

		if phip>0.0 and phis>0.0:
			# print 'Hurry! phip is:', phip, 'and phis is:', phis
			pass
			# break

	if phip==0.0 or phis==0.0:
		print 'Program Failed to get value of phip and phis'
		# raise ValueError('Program Failed to get value of phip and phis')

	# phip,phis = nsolve([EOS_m,(mu_s_m-mu_s0.subs(phi_s,phis0))],[phi_p,phi_s],[0.85,0.05],verify=True)

	# print phip_all_values
	# print phis_all_values
	
	phip_all_values = npy.array(remove_duplicates(phip_all_values))
	phis_all_values = npy.array(remove_duplicates(phis_all_values))

	print phip_all_values
	print phis_all_values

	for i in range(len(phip_all_values)-1):
		if phip_all_values[i+1]>phip_all_values[i]:
			phip = phip_all_values[i+1]
			phis = phis_all_values[i+1]
		else:
			phip = phip_all_values[i]
			phis = phis_all_values[i]		

	print 'Is phip complex:', phip
	print 'Is phis complex:', phis
	
	# phip=abs(phip)
	# phis=abs(phis)

	#CHECKING IF MIXTURE VOLUME FRACTION RESULTS ARE VALID.
	checkVolumeFraction([phip,phis],['phi_p','phi_s'])
	
	#PRINTING OF RESULTS OF MIXTURE COMPOSITION CALCULATIONS.
	#FOR DIAGNOSTIC PURPOSES.
	if verbose:
		print('phip = {}, phis = {};'.format(phip,phis))
	# phip0=0.0 #junk
	return [P,T,phip,phis,phip0,phis0]

def binarySolubilitySwellingCHV(P,T,Mp,Ms,**kwargs):
	# print 'this is also a great great great problem'
	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))
	
	if 'alpha_p' in kwargs and 'vhp' in kwargs and 'epsilon_p' in kwargs:
		Ppstar,Tpstar,Vpstar = calculateCharacteristicParameters(alpha_p,vhp,epsilon_p,Mp)
	elif 'Ppstar' in kwargs and 'Tpstar' in kwargs and 'Rpstar' in kwargs:
		pass
	else:
		raise ValueError('In binarySolubilitySwellingCHV, polymer parameters: Either molecular (alpha_p,vhp,epsilon_p) or characteristic (Ppstar,Tpstar,Rpstar) parameters must be passed into keyword arguments.')
	
	if 'alpha_s' in kwargs and 'vhs' in kwargs and 'epsilon_s' in kwargs:
		Psstar,Tsstar,Vsstar = calculateCharacteristicParameters(alpha_s,vhs,epsilon_s,Ms)
	elif 'Psstar' in kwargs and 'Tsstar' in kwargs and 'Rsstar' in kwargs:
		pass
	else:
		raise ValueError('In binarySolubilitySwellingCHV, solvent parameters: Either molecular (alpha_s,vhs,epsilon_s) or characteristic (Psstar,Tsstar,Rsstar) parameters must be passed into keyword arguments.')
	
	if 'k12' in kwargs and 'delta' in kwargs:
		pass
	elif 'zeta' in kwargs and 'delta' in kwargs:
		pass
	else:
		raise ValueError('In binarySolubilitySwellingCHV, mixture parameters: (k12,delta) or (zeta,delta) mixture parameters must be passed into keyword arguments.')
	
	#Boolean determining whether information is printed.
	#Default option is False.
	verbose = kwargs.get('verbose',False)
	
	# Boolean that determines method of calculation.
	#	 True: Uses simplified (original) swelling calculation assuming pure polymer.
	#	 False: Uses more sophisticated swelling calculation assuming air (N2) content.
	simplified = kwargs.pop('simplified',True)
	
	#PURE FLUID PARAMETERS.
	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar
	alpha_p0 = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_s0 = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	
	#MIXTURE PARAMETERS.
	vhm = delta*vhs
	alpha_p = alpha_p0*vhp/vhm
	alpha_s = alpha_s0*vhs/vhm
	
	# CALCULATION OF VOLUME FRACTIONS AT P, T.
	if verbose:
		print('High-pressure solvent environment:')
	[Pd,Td,hsol_phip,hsol_phis,hphip0,hphis0] = binaryPhaseEquilibriumCHV(P,T,Mp,Ms,**kwargs)
	
	#CALCULATION OF SOLVENT SOLUBILITY (MASS FRACTION) AT P, T.
	ms = (Ms*hsol_phis/alpha_s)/(Mp*hsol_phip/alpha_p+Ms*hsol_phis/alpha_s)   #Kier Original
	# ms = (Ms*hsol_phis/alpha_s0)/(Mp*hsol_phip/alpha_p0+Ms*hsol_phis/alpha_s0)   #Condo Solubility
	Sw = hphip0/hsol_phip
	
	#PRINTING OF RESULTS OF SOLUBILITY AND SWELLING.
	#FOR DIAGNOSTIC PURPOSES.
	if verbose:
		print('ms = {}, Sw = {};'.format(ms,Sw))

	
	#PRINTING OF RESULTS OF SOLUBILITY AND SWELLING.
	print('At P = {}, T = {}, zeta = {}, delta = {};'.format(P,T,zeta,delta))
	print('Xs = {}, Sw = {};'.format(ms,Sw))
	print('phip = {}, phis = {}, phip0 = {}, phis0 = {};'.format(hsol_phip,hsol_phis,hphip0,hphis0))

	Rtilde=hsol_phip+hsol_phis

	return [P,T,ms,Sw,hsol_phip,hsol_phis,Rtilde,hphip0,hphis0]

def calculateBinarySolubilitySwelling(theory,P0,T0,Mp,Ms,**kwargs):

	if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
		exec "XSw = binarySolubilitySwelling%s(P0,T0,Mp,Ms,**kwargs)" % (theory)
		result = XSw
	
	elif not isListOrNpyArray(T0) and isListOrNpyArray(P0):
		result = [[range(0,len(P0))] for x in range(9)]
		T = range(0,len(P0))
		m_s = range(0,len(P0))
		Sw = range(0,len(P0))
		phip = range(0,len(P0))
		phis = range(0,len(P0))
		Rtilde = range(0,len(P0))
		phip0 = range(0,len(P0))
		phis0 = range(0,len(P0))
		
		for i in range(0,len(P0)):
			exec "XSw = binarySolubilitySwelling%s(P0[i],T0,Mp,Ms,**kwargs)" % (theory)
			T[i] = XSw[1]
			m_s[i] = XSw[2]
			Sw[i] = XSw[3]
			phip[i]=XSw[4]
			phis[i]=XSw[5]
			Rtilde[i]=XSw[6]
			phip0[i]=XSw[7]
			phis0[i]=XSw[8]

		result[0] = P0
		result[1] = T
		result[2] = m_s
		result[3] = Sw
		result[4]=phip
		result[5]=phis
		result[6]=Rtilde
		result[7]=phip0
		result[8]=phis0

	elif not isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(9)]
		P = range(0,len(T0))
		m_s = range(0,len(T0))
		Sw = range(0,len(T0))
		phip = range(0,len(T0))
		phis = range(0,len(T0))
		Rtilde = range(0,len(T0))
		phip0 = range(0,len(T0))
		phis0 = range(0,len(T0))

		for i in range(0,len(T0)):
			exec "XSw = binarySolubilitySwelling%s(P0,T0[i],Mp,Ms,**kwargs)" % (theory)
			P[i] = XSw[0]
			m_s[i] = XSw[2]
			Sw[i] = XSw[3]
			phip[i]=XSw[4]
			phis[i]=XSw[5]
			Rtilde[i]=XSw[6]
			phip0[i]=XSw[7]
			phis0[i]=XSw[8]
	
		result[0] = P
		result[1] = T0
		result[2] = m_s
		result[3] = Sw
		result[4]=phip
		result[5]=phis
		result[6]=Rtilde
		result[7]=phip0
		result[8]=phis0
	
	elif isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(9)]
		P = range(0,len(T0))
		m_s = range(0,len(T0))
		Sw = range(0,len(T0))
		phip = range(0,len(T0))
		phis = range(0,len(T0))
		Rtilde = range(0,len(T0))
		phip0 = range(0,len(T0))
		phis0 = range(0,len(T0))

		for i in range(0,len(T0)):
			exec "XSw = binarySolubilitySwelling%s(P0[i],T0[i],Mp,Ms,**kwargs)" % (theory)
			P[i] = XSw[0]
			m_s[i] = XSw[2]
			Sw[i] = XSw[3]
			phip[i]=XSw[4]
			phis[i]=XSw[5]
			Rtilde[i]=XSw[6]
			phip0[i]=XSw[7]
			phis0[i]=XSw[8]

		result[0] = P0
		result[1] = T0
		result[2] = m_s
		result[3] = Sw
		result[4]=phip
		result[5]=phis
		result[6]=Rtilde
		result[7]=phip0
		result[8]=phis0
	
	else:
		raise ValueError('In calculateBinarySwelling: Unknown error involving P0 and T0.')
	
	return result

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
# v_0,alpha,B0,B1 = Tait_Parameters_of_Different_Polymers(**kwargs)

number_of_isotherm, result = Split_Isotherms(P0_X,T0_X,X0_X,'X')
P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5,P0_X_T6,T0_X_T6,X0_X_T6,P0_X_T7,T0_X_T7,X0_X_T7,P0_X_T8,T0_X_T8,X0_X_T8,P0_X_T9,T0_X_T9,X0_X_T9 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23],result[24],result[25],result[26]
# print P0_X_T1,T0_X_T1,X0_X_T1,P0_X_T2,T0_X_T2,X0_X_T2,P0_X_T3,T0_X_T3,X0_X_T3,P0_X_T4,T0_X_T4,X0_X_T4,P0_X_T5,T0_X_T5,X0_X_T5

number_of_isotherm_swelling, result = Split_Isotherms(P0_S,T0_S,S0_S,'S')
P0_S_T1,T0_S_T1,S0_S_T1,P0_S_T2,T0_S_T2,S0_S_T2,P0_S_T3,T0_S_T3,S0_S_T3,P0_S_T4,T0_S_T4,S0_S_T4,P0_S_T5,T0_S_T5,S0_S_T5,P0_S_T6,T0_S_T6,S0_S_T6,P0_S_T7,T0_S_T7,S0_S_T7,P0_S_T8,T0_S_T8,S0_S_T8,P0_S_T9,T0_S_T9,S0_S_T9 = result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],result[19],result[20],result[21],result[22],result[23],result[24],result[25],result[26]
# print P0_S_T1,T0_S_T1,S0_S_T1,P0_S_T2,T0_S_T2,S0_S_T2,P0_S_T3,T0_S_T3,S0_S_T3,P0_S_T4,T0_S_T4,S0_S_T4,P0_S_T5,T0_S_T5,S0_S_T5

# P0_S_T2,T0_S_T2,S0_S_T2 = P0_S_T1,T0_S_T1,S0_S_T1
# P0_S_T1,T0_S_T1,S0_S_T1 =  P0_S_T5,T0_S_T5,S0_S_T5

Kier=False
Hassan=True  
Hassan_Var_Vol=False  
Condo=False  
Condo_Original=False 

kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight,'Kier':Kier,'Hassan':Hassan,'Hassan_Var_Vol':Hassan_Var_Vol,'Condo':Condo,'Condo_Original':True}

cepsilon_s,cepsilon_p,cz,czeta,epsilon_p,g,x,delta,zeta=Parameters_for_Mixtures_and_Tg(**kwargs)
cdelta=100.0

xS_infty=x*(Ppstar/(Tpstar*Rpstar))*(1+ln(1+g))
# print Ppstar,Tpstar,Rpstar,g,epsilon_p,x,xS_infty

#Paper12 and Paper13 Are Working Best Simultaneously
# zeta=	1.07136875	#0.95125453	#1.16681479	#1.09730204	#1.04532004	#1.16681479	#1.08884102	#1.04185590	#1.10678999	#1.03166800	#0.96584091	#1.07703406	#0.96584091	#1.04185590	#1.05128789	#0.93486423	#0.94730914
# delta=	1.23734185	#0.96557611	#1.46977534	#1.30622455	#1.19773798	#1.46977534	#1.29082731	#1.18464299	#1.33805847	#1.16318167	#1.04196138	#1.26576563	#1.04196138	#1.18464299	#1.19048236	#0.96152684	#0.98671535

#===================================================================================
#Determining theoretical solubility/swelling.
#===================================================================================

print('DHV mixture parameters zeta = {} and delta = {}.'.format(zeta,delta))

gammas,vhs,epsilons = calculateNewMolecularParameters(Psstar,Tsstar,Rsstar,Ms)
vh = delta*vhs/NA
print('The hole volume is vh = {}.'.format(vh))

Pmin = min(P0_X)
Pmax = max(P0_X)
Tmin = min(T0_X)
Tmax = max(T0_X)

print('The pressure range is {}-{}MPa and the temperature range is {}-{} K.'.format(Pmin,Pmax,Tmin,Tmax))

# P0_X_T4 = npy.concatenate(([0.0001],P0_X_T4),axis=0)
# T0_X_T4 = npy.concatenate(([373.15],T0_X_T4),axis=0)
# X0_X_T4 = npy.concatenate(([0],X0_X_T4),axis=0)

# P0_S_T4 = npy.concatenate(([0.0001],P0_S_T4),axis=0)
# T0_S_T4 = npy.concatenate(([373.15],T0_S_T4),axis=0)
# S0_S_T4 = npy.concatenate(([1.0],S0_S_T4),axis=0)
'''
P0 = npy.linspace(min(P0_X),max(P0_X),15)
# P0 = npy.linspace(2.0,17,15)
T1=T0_X_T1[0]	#403	#290
T2=T0_X_T2[0]	#423	#304
T3=T0_X_T3[0]	#463	#350
T4=T0_X_T4[0]	#423	#304
T5=T0_X_T5[0]	#463	#350
T6=0.0#T0_X_T6[0]	#423	#304
T7=0.0#T0_X_T7[0]	#463	#350
T8=0.0#T0_X_T8[0]	#463	#350
T9=0.0#T0_X_T9[0]	#463	#350

number_of_points = 15
P1 = npy.linspace(min(P0_X),max(P0_X_T1),number_of_points)
P2 = npy.linspace(min(P0_X),max(P0_X_T2),number_of_points)
P3 = npy.linspace(min(P0_X),max(P0_X_T3),number_of_points)
P4 = npy.linspace(min(P0_X),max(P0_X_T4),number_of_points)
P5 = npy.linspace(min(P0_X),max(P0_X_T5),number_of_points)
'''
'''
verbose = False

if T1!=0.0:
	result = calculateBinarySolubilitySwelling('CHV',P1,T1,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T1_DHV = result[2]
	Sw_T1_DHV = result[3]
if T2!=0.0:
	result = calculateBinarySolubilitySwelling('CHV',P2,T2,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T2_DHV = result[2]
	Sw_T2_DHV = result[3]
if T3!=0.0:
	result = calculateBinarySolubilitySwelling('CHV',P3,T3,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T3_DHV = result[2]
	Sw_T3_DHV = result[3]
if T4!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P4,T4,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T4_DHV = result[2]	
	Sw_T4_DHV = result[3]
if T5!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P5,T5,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T5_DHV = result[2]	
	Sw_T5_DHV = result[3]
if T6!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P6,T6,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T6_DHV = result[2]	
	Sw_T6_DHV = result[3]
if T7!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P7,T7,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T7_DHV = result[2]	
	Sw_T7_DHV = result[3]
if T8!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P8,T8,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T8_DHV = result[2]	
	Sw_T8_DHV = result[3]
if T9!=0.0:	
	result = calculateBinarySolubilitySwelling('CHV',P9,T9,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',verbose=verbose)
	Xs_T9_DHV = result[2]	
	Sw_T9_DHV = result[3]
'''

Isobars = True
verbose = False
Entropy = True
Plot_S2 = False
plot_phi = True

if Isobars:
	number_of_isobar=3
	# T0 = npy.linspace(min(T0_X),max(T0_X),10)
	T0 = npy.linspace(200,400,15)		#max: 1400000  #Small pressure ==> entropy max reaches at smaller temperature
	P1=3.0#P0_X_P1[0]#0.101325
	P2=0.0#30.0
	P3=0.0#50.0

if P1!=0.0:
	result = calculateBinarySolubilitySwelling('CHV',P1,T0,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate')
	Xs_P1_DHV = result[2]
	Sw_P1_DHV = result[3]
	phip_P1_DHV = result[4]
	phis_P1_DHV = result[5]
	Rtilde_P1_DHV = result[6]
	phip0_P1_DHV = result[7]
	phis0_P1_DHV = result[8]
	
	if Entropy:
		properties=calculateThermodynamicVariables(P1,T0,phip_P1_DHV,phis_P1_DHV,phip0_P1_DHV,phis0_P1_DHV,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
		S_1_P1_DHV = properties[2]
		S_2_P1_DHV = properties[3]

if P2!=0.0:
	result = calculateBinarySolubilitySwelling('CHV',P2,T0,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate')
	Xs_P2_DHV = result[2]
	Sw_P2_DHV = result[3]
	phip_P2_DHV = result[4]
	phis_P2_DHV = result[5]
	Rtilde_P2_DHV = result[6]
	phip0_P2_DHV = result[7]
	phis0_P2_DHV = result[8]
	
	if Entropy:
		properties=calculateThermodynamicVariables(P2,T0,phip_P2_DHV,phis_P2_DHV,phip0_P2_DHV,phis0_P2_DHV,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
		S_1_P2_DHV = properties[2]
		S_2_P2_DHV = properties[3]

if P3!=0.0:
	result = calculateBinarySolubilitySwelling('CHV',P3,T0,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate')
	Xs_P3_DHV = result[2]
	Sw_P3_DHV = result[3]
	phip_P3_DHV = result[4]
	phis_P3_DHV = result[5]
	Rtilde_P3_DHV = result[6]
	phip0_P3_DHV = result[7]
	phis0_P3_DHV = result[8]
	
	if Entropy:
		properties=calculateThermodynamicVariables(P3,T0,phip_P3_DHV,phis_P3_DHV,phip0_P3_DHV,phis0_P3_DHV,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
		S_1_P3_DHV = properties[2]
		S_2_P3_DHV = properties[3]

#===================================================================================
#Plotting the PMMA/CO2 mixture results.
#===================================================================================

#Setting saved image properties
img_extension = '.png'
img_dpi = None
output_folder = 'Post_Thesis'

#Checking for existence of output directory. If such a directory doesn't exist, one is created.
if not os.path.exists('./'+output_folder):
    os.makedirs('./'+output_folder)

#Setting font size
axis_size = 20
title_size = 20
size = 14
label_size = 20

plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size

#Defining axes
solubility_axes = [7.5,42.5,0.0,0.14]
swelling_axes = [5,25,1.0,1.17]
# phi_axes = [5,25,0.0,1.0]
# TD_axes = [5,25,0.0,5.0]

#Markers
mark1 = 'o'
mark2 = 's'
mark3 = '>'
mark4 = '<'
mark5 = 'D'
mark6 = 'H'
mark7 = 'P'
mark8 = 'X'


linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

#Linestyles
ls1 = '-'
ls2 = '--'
ls3 = ':'
ls4 = '-.'
ls5 = linestyles['densely dashdotdotted']

#General line properties.
linewidth = 2
markersize = 8

'''
#Plotting the solubility of the PMMA+CO2 mixture.
figX = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

if T1!=0.0:
	plt.plot(P0_X_T1,X0_X_T1,'k',marker=mark1,ls='',label='Experiment at {} K'.format(T0_X_T1[0]),ms=markersize)
if T2!=0.0:
	plt.plot(P0_X_T2,X0_X_T2,'k',marker=mark2,ls='',label='{} K'.format(T0_X_T2[0]),ms=markersize)
if T3!=0.0:
	plt.plot(P0_X_T3,X0_X_T3,'k',marker=mark3,ls='',label='{} K'.format(T0_X_T3[0]),ms=markersize)
if T4!=0.0:
	plt.plot(P0_X_T4,X0_X_T4,'k',marker=mark4,ls='',label='{} K'.format(T0_X_T4[0]),ms=markersize)
if T5!=0.0:
	plt.plot(P0_X_T5,X0_X_T5,'k',marker=mark5,ls='',label='{} K'.format(T0_X_T5[0]),ms=markersize)
if T6!=0.0:
	plt.plot(P0_X_T6,X0_X_T6,'y',marker=mark6,ls='',label='{} K'.format(T0_X_T6[0]),ms=markersize)
if T7!=0.0:
	plt.plot(P0_X_T7,X0_X_T7,'r',marker=mark7,ls='',label='{} K'.format(T0_X_T7[0]),ms=markersize)
if T8!=0.0:
	plt.plot(P0_X_T8,X0_X_T8,'b',marker=mark8,ls='',label='{} K'.format(T0_X_T8[0]),ms=markersize)
if T9!=0.0:
	plt.plot(P0_X_T9,X0_X_T9,'g',marker=mark1,ls='',label='{} K'.format(T0_X_T9[0]),ms=markersize)
						
if T1!=0.0:
	plt.plot(P1,Xs_T1_DHV,'k',ls=ls1,label='Present theory {} K'.format(T1),lw=linewidth)
if T2!=0.0:
	plt.plot(P2,Xs_T2_DHV,'k',ls=ls2,label='{} K'.format(T2),lw=linewidth)
if T3!=0.0:
	plt.plot(P3,Xs_T3_DHV,'k',ls=ls3,label='{} K'.format(T3),lw=linewidth)
if T4!=0.0:
	plt.plot(P4,Xs_T4_DHV,'k',ls=ls4,label='{} K'.format(T4),lw=linewidth)
if T5!=0.0:
	plt.plot(P5,Xs_T5_DHV,'k',ls=ls5,label='{} K'.format(T5),lw=linewidth)
if T6!=0.0:
	plt.plot(P6,Xs_T6_DHV,'y',ls=ls3,label='{} K'.format(T6),lw=linewidth)
if T7!=0.0:
	plt.plot(P7,Xs_T7_DHV,'r',ls=ls3,label='{} K'.format(T7),lw=linewidth)
if T8!=0.0:
	plt.plot(P8,Xs_T8_DHV,'b',ls=ls3,label='{} K'.format(T8),lw=linewidth)
if T9!=0.0:
	plt.plot(P9,Xs_T9_DHV,'g',ls=ls3,label='{} K'.format(T9),lw=linewidth)

plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
plt.ylabel('Gas solubility $\chi_s$ (g/g)',fontsize=axis_size)
plt.legend(loc=2,fontsize=size,numpoints=1)
# plt.title(kwargs, fontdict=None, loc='center', pad=None)
# plt.axis(solubility_axes)
plt.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.10,wspace=0.30,hspace=0.25)
figX.savefig('./'+output_folder+r'\PS_CO2_Self_Grassia_02kilo_POST_THESIS_Paper4_11_12_Solubility'+img_extension,dpi=240)

#Plotting the swelling of the PMMA+CO2 mixture.
figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
if T1!=0.0:
	plt.plot(P0_S_T1,S0_S_T1,'k',marker=mark1,ls='',label='Experiment at {} K'.format(T0_S_T1[0]),ms=markersize)
if T2!=0.0:
	plt.plot(P0_S_T2,S0_S_T2,'k',marker=mark3,ls='',label='{} K'.format(T0_S_T2[0]),ms=markersize)
if T3!=0.0:
	plt.plot(P0_S_T3,S0_S_T3,'k',marker=mark4,ls='',label='{} K'.format(T0_S_T3[0]),ms=markersize)
if T4!=0.0:
	plt.plot(P0_S_T4,S0_S_T4,'k',marker=mark5,ls='',label='{} K'.format(T0_S_T4[0]),ms=markersize)
# if T5!=0.0:
# 	plt.plot(P0_S_T5,S0_S_T5,'k',marker=mark5,ls='',label='{} K'.format(T0_S_T5[0]),ms=markersize)
if T6!=0.0:
	plt.plot(P0_S_T6,S0_S_T6,'y',marker=mark6,ls='',label='{} K'.format(T0_S_T6[0]),ms=markersize)
if T7!=0.0:
	plt.plot(P0_S_T7,S0_S_T7,'r',marker=mark7,ls='',label='{} K'.format(T0_S_T7[0]),ms=markersize)
if T8!=0.0:
	plt.plot(P0_S_T8,S0_S_T8,'b',marker=mark8,ls='',label='{} K'.format(T0_S_T8[0]),ms=markersize)
if T9!=0.0:
	plt.plot(P0_S_T9,S0_S_T9,'g',marker=mark4,ls='',label='{} K'.format(T0_S_T9[0]),ms=markersize)

if T1!=0.0:
	plt.plot(P1,Sw_T1_DHV,'k',ls=ls1,label='Present theory {} K'.format(T1),lw=linewidth)
if T2!=0.0:
	plt.plot(P2,Sw_T2_DHV,'k',ls=ls2,label='{} K'.format(T2),lw=linewidth)
if T3!=0.0:
	plt.plot(P3,Sw_T3_DHV,'k',ls=ls3,label='{} K'.format(T3),lw=linewidth)
if T4!=0.0:
	plt.plot(P4,Sw_T4_DHV,'k',ls=ls4,label='{} K'.format(T4),lw=linewidth)
if T5!=0.0:
	plt.plot(P5,Sw_T5_DHV,'k',ls=ls5,label='{} K'.format(T5),lw=linewidth)
if T6!=0.0:
	plt.plot(P6,Sw_T6_DHV,'y',ls=ls3,label='{} K'.format(T6),lw=linewidth)
if T7!=0.0:
	plt.plot(P7,Sw_T7_DHV,'r',ls=ls3,label='{} K'.format(T7),lw=linewidth)
if T8!=0.0:
	plt.plot(P8,Sw_T8_DHV,'b',ls=ls3,label='{} K'.format(T8),lw=linewidth)
if T9!=0.0:
	plt.plot(P9,Sw_T9_DHV,'g',ls=ls3,label='{} K'.format(T9),lw=linewidth)

plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
plt.ylabel('Swelling ratio $S_W$ $(cm^3/cm^3)$',fontsize=axis_size)
plt.legend(loc=2,fontsize=size,numpoints=1)
# plt.title(kwargs, fontdict=None, loc='center', pad=None)
# plt.axis(swelling_axes)
plt.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.10,wspace=0.30,hspace=0.25)
figS.savefig('./'+output_folder+r'\PS_CO2_Self_Grassia_02kilo_POST_THESIS_Paper4_11_12_Swelling'+img_extension,dpi=240)
'''

	
if plot_phi:

	#Plotting the phi's of the PS+CO2 mixture.
	figphi = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

	if P1!=0.0:	
		plt.plot(T0,phip_P1_DHV,'r',ls=ls1,label='phip_{} MPa'.format(P1),lw=linewidth)
		plt.plot(T0,phis_P1_DHV,'r',ls=ls2,label='phis_{} MPa'.format(P1),lw=linewidth)
		plt.plot(T0,phip0_P1_DHV,'r',ls=ls3,label='phip0_{} MPa'.format(P1),lw=linewidth)
		plt.plot(T0,phis0_P1_DHV,'r',ls=ls4,label='phis0_{} MPa'.format(P1),lw=linewidth)
	if P2!=0.0:
		plt.plot(T0,phip_P2_DHV,'b',ls=ls1,label='phip_{} MPa'.format(P2),lw=linewidth)
		plt.plot(T0,phis_P2_DHV,'b',ls=ls2,label='phis_{} MPa'.format(P2),lw=linewidth)
		plt.plot(T0,phip0_P2_DHV,'b',ls=ls3,label='phip0_{} MPa'.format(P2),lw=linewidth)
		plt.plot(T0,phis0_P2_DHV,'b',ls=ls4,label='phis0_{} MPa'.format(P2),lw=linewidth)
	if P3!=0.0:
		plt.plot(T0,phip_P3_DHV,'g',ls=ls1,label='phip_{} MPa'.format(P3),lw=linewidth)
		plt.plot(T0,phis_P3_DHV,'g',ls=ls2,label='phis_{} MPa'.format(P3),lw=linewidth)
		plt.plot(T0,phip0_P3_DHV,'g',ls=ls3,label='phip0_{} MPa'.format(P3),lw=linewidth)
		plt.plot(T0,phis0_P3_DHV,'g',ls=ls4,label='phis0_{} MPa'.format(P3),lw=linewidth)

	plt.xlabel('Temperature T (K)',fontsize=axis_size)
	plt.ylabel('phi',fontsize=axis_size)
	plt.legend(loc=4,fontsize=size,numpoints=1)
	plt.title(kwargs, fontdict=None, loc='center', pad=None)
	# plt.axis(TD_axes)
	# figphi.savefig('./'+output_folder+r'\bin_PS_CO2_phi'+img_extension,dpi=img_dpi)


if Entropy:
	#Plotting the phi's of the PS+CO2 mixture.
	figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

	plt.plot(T0,S_1_P1_DHV,'r',ls=ls1,label='S_1_{} MPa'.format(P1),lw=linewidth)
	if P2!=0.0:
		plt.plot(T0,S_1_P2_DHV,'r',ls=ls2,label='S_1_{} MPa'.format(P2),lw=linewidth)
	if P3!=0.0:
		plt.plot(T0,S_1_P3_DHV,'r',ls=ls3,label='S_1_{} MPa'.format(P3),lw=linewidth)

	if Plot_S2:
		
		plt.plot(T0,S_2_P1_DHV,'m',ls=ls1,label='S_2_{} MPa'.format(P1),lw=linewidth)
		if P2!=0.0:
			plt.plot(T0,S_2_P2_DHV,'m',ls=ls2,label='S_2_{} MPa'.format(P2),lw=linewidth)
		if P3!=0.0:
			plt.plot(T0,S_2_P3_DHV,'m',ls=ls3,label='S_2_{} MPa'.format(P3),lw=linewidth)
	
	# plt.axvline(x=378,lw=0.5,color='k', linestyle='-.')
	# plt.axhline(y=0.0,lw=0.5,color='k', linestyle='-.')
	# S_max=npy.max(S_1_P1)
	# print 'S_max is:', S_max
	Tg_line=xS_infty#0.317*0.98268 #0.310707*0.8708171#0.2361#2.38*0.271##0.2558 #S_max*kx #0.2361
	plt.axhline(y=Tg_line,lw=0.5,color='k', linestyle='-.')

	plt.xlabel('Temperature T (K)',fontsize=axis_size)
	plt.ylabel('Entropy',fontsize=axis_size)
	plt.legend(loc=4,fontsize=size,numpoints=1)
	plt.title(kwargs, fontdict=None, loc='center', pad=None)
	# plt.axis(TD_axes)
	# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

#Show plot windows.
plt.show()