# Author: Hassan Alam (Modified Kier von Konigslow Code)

import os,sys,math,csv,matplotlib.pyplot as plt,numpy as npy
from loadPhysicalConstants import *
from checkResults import *
from sympy import *
import warnings
from loadExperimentalData import *
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from calculatePureVariables import calculateNewMolecularParameters
# from calculateBinaryResidual import calculateBinarySSQ
# from calculateBinaryVariablesCHV import *
from Parameters_of_Different_Polymers import *
from Parameters_for_Mixtures_and_Tg import *
# from All_Functions import *
from SplittingExperimentalDataOf_X_Sw_aboveANDbelowTgANDIsotherms import *
# from Tait_Parameters_of_Different_Polymers import *
# from loadExperimentalDataCO2 import *
# from CO2PVT_interpolation import *
from Split_Exp_Data_in_Isotherms import*
from collections import OrderedDict			#For Exotic Line Styles


def remove_duplicates(lst):
    res = []
    for x in lst:
        if x not in res:
            res.append(x)
    return res

def remove_two_lists_simultaneous_duplicates(lst1,lst2):

    res1 = []
    res2 = []

    for i in range(len(lst1)):
        if (lst1[i] not in res1) or (lst2[i] not in res2):
            res1.append(lst1[i])
            res2.append(lst2[i])

    return res1,res2

def binaryPhaseEquilibriumCondo_Original_nsolve(P,T,Mp,Ms,**kwargs):

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

	#Allows for method argument in kwargs. Options are: 'disparate', 'single', 'mixed'.
	#Default option is 'disparate'.
	# -'disparate'	--> Mixture phase has constant hole volume of vhm. Pure phases have constant hole volumes vhp, vhs.
	# -'single'		--> All phases have hole volume vhm.
	# method = kwargs.pop('method','disparate')
	
	#Boolean determining whether information is printed.
	#Default option is False.
	verbose = kwargs.pop('verbose',False)
	if verbose:
		print('FOR: P = {}MPa, T = {}K;'.format(P,T))
	
	#Initializing phi_p, phi_s and v_h as symbolic variables for the sympy package.
	#This is a step necessary for the numerical solver nsolve.
	# phi_p = Symbol('phi_p',real=True)
	# v_h = Symbol('v_h',real=True)
	Rtilde_ = Symbol('Rtilde_',real=True)	
	phi_s = Symbol('phi_s',real=True)

	#PURE FLUID PARAMETERS.
	rs = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	rp = (Ppstar*Mp)/(kB*Tpstar*Rpstar)

	#PURE FLUID PARAMETERS.
	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar

	Tspstar=zeta*math.sqrt(Tsstar*Tpstar)
	Xsp=(Tsstar+Tpstar-2*Tspstar)/T

	Pstilde=P/Psstar
	Tstilde=T/Tsstar
	Pptilde=P/Ppstar
	Tptilde=T/Tpstar

	# print Psstar
	# print Tsstar
	# print Rsstar
	# print rs
	# print Ms
	# print zeta
	# print Ppstar
	# print Tpstar
	# print Rpstar
	# print rp
	# print Mp

	phip=1-phi_s
	vhm=phi_s*vhs+phip*vhp
	Tstar=phi_s*Tsstar+phip*Tpstar-phi_s*phip*T*Xsp
	Pstar=kB*Tstar/vhm
	r=1/(phi_s/rs+phip/rp)

	Ptilde=P/Pstar
	Ttilde=T/Tstar
	vtilde=1/Rtilde_

	# print 'rrrrrrrrrrrrrrrrr', Ptilde
	# print 'eeeeeeeeeeeeeeeee', Ttilde

	EOS=Rtilde_**2+Ptilde+Ttilde*(ln(1-Rtilde_)+(1-1/r)*Rtilde_)
	# EOS = v_h*P/(kB*T)-(chi_pp/2)*phi_p**2-chi_ps*phi_p*phi_s-(chi_ss/2)*phi_s**2+(1.0-1.0/alpha_p)*phi_p+(1.0-1.0/alpha_s)*phi_s+log(1.0-phi_p-phi_s)
	#Mixture equation of state in general.
	EOS_m = EOS.subs([(Rtilde_,Rtilde_),(phi_s,phi_s)])
	
	#Mixture solvent chemical potential.
	mu_s = (ln(phi_s)+(1-rs/rp)*phip+rs*Rtilde_*Xsp*phip**2+rs*((-1*Rtilde_+Pstilde*vtilde)/Tstilde+(vtilde-1)*ln(1-Rtilde_)+ln(Rtilde_)/rs))
	# mu_s = alpha_s*(chi_ss*phi_s+chi_ps*phi_p+(1.0/alpha_s)*(1+log(phi_s))-log(1-phi_p-phi_s)-1)  #Hassan: This is my correction.

	#Mixture solvent chemical potential in general.
	mu_s_m = mu_s.subs([(Rtilde_,Rtilde_),(phi_s,phi_s)])

	#Mixture equation of state in phi_s --> 0 limit. v_h takes on value vhp.
	EOS_p0 = EOS.subs([(Rtilde_,Rtilde_),(phi_s,0.0)])
	#Mixture equation of state in phi_p --> 0 limit. v_h takes on value vhs.
	EOS_s0 = EOS.subs([(Rtilde_,Rtilde_),(phi_s,1.0)])
	#Mixture solvent chemical potential in phi_p --> 0 limit. v_h takes on value vhs.
	mu_s0 = mu_s.subs([(Rtilde_,Rtilde_),(phi_s,1.0)])

	print('P = {}, T = {};'.format(P,T))

	#CALCULATION OF PURE FLUID STATE AT P, T.
	#Solving for the volume fraction of the system in the pure fluid limiting cases.
	Rtildep0 = nsolve(EOS_p0,Rtilde_,0.97,verify=True)		#0.67
	
	Try_all_solvent_guesses = True
	
	if Try_all_solvent_guesses == False:
		Rtildes0 = nsolve(EOS_s0,Rtilde_,0.01,verify=True)		#0.1
	elif Try_all_solvent_guesses == True:
		Rtildes0 = 0.0
		Rtildes0_all_values = []

		# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.000001,0.00001,0.90,0.95,0.97,0.98,0.99,0.999,0.9999,0.99999,0.999999])
		# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.65,0.75,0.85,0.00001,0.90,0.97,0.99,0.999,0.9999])
		# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.30,0.45,0.70,0.90,0.95,0.999])
		# guess2 = npy.array([0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090,0.100,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095])
		guess2 = npy.array([0.65,0.85,0.95,0.45,0.30,0.01,0.10,0.05,0.03,0.02])
		# guess2 = npy.array([0.65,0.85,0.95,0.45,0.30])
		guess_low = ((Tstilde/rs)-sqrt((Tstilde/rs)**2-4*(1-(Tstilde/2))*Pstilde))/(2*(1-(Tstilde/2)))  #Correct Guess by quadratic approximation for smallest Rstilde0
		# print guess_low
		# guess2 = npy.array([guess_low,guess_low+0.01,guess_low-0.01,0.010,0.020,0.030,0.040])
		# coeff = [(Tstilde/3), ((Tstilde/2)-1), (Tstilde/rs), -Pstilde]			#Correct Guess by cubic approximation for smallest Rstilde0
		# coeff = [(Tstilde/4), (Tstilde/3), ((Tstilde/2)-1), (Tstilde/rs), -Pstilde]			#Correct Guess by forth power approximation for smallest Rstilde0
		# coeff = [(Tstilde/5), (Tstilde/4), (Tstilde/3), ((Tstilde/2)-1), (Tstilde/rs), -Pstilde]			#Correct Guess by fifth power approximation for smallest Rstilde0
		# coeff = [(Tstilde/6), (Tstilde/5), (Tstilde/4), (Tstilde/3), ((Tstilde/2)-1), (Tstilde/rs), -Pstilde]			#Correct Guess by sixth power approximation for smallest Rstilde0
		coeff = [(Tstilde/7), (Tstilde/6), (Tstilde/5), (Tstilde/4), (Tstilde/3), ((Tstilde/2)-1), (Tstilde/rs), -Pstilde]			#Correct Guess by seventh power approximation for smallest Rstilde0

		root_final = 999.0
		answer = npy.roots(coeff)
		# print answer
		for i in range(len(answer)):
			root = complex(answer[i])
			if root.real>0.0 and abs(root.imag)<=10E-3:
				if root_final>root.real:
					root_final = root.real
					picked = root_final
		Rtildes0_all_values.append(picked)
		# print 'Rtildes0_all_values picked from cubic approximation is:', Rtildes0_all_values

		for i in range(len(guess2)):
			# print 'for loop of Rtildes0:', i, 'Rtildes0 of last loop is:', Rtildes0
			try:
				Rtildes0 = nsolve(EOS_s0,Rtilde_,guess2[i],verify=True)
				# print Rtildes0, T
			except:
				pass

			Rtildes0 = complex(Rtildes0)

			if Rtildes0.real>0.0 and abs(Rtildes0.imag)<=10E-3:
				# print 'Is Rtildes0 complex:',Rtildes0
				Rtildes0 = abs(Rtildes0)
				Rtildes0_all_values.append(Rtildes0)
				# break			#Do not break it because CO2 has multiple solution.
			else:
				# Rtildes0 = 0.0
				pass

		Rtildes0_all_values = npy.array(remove_duplicates(Rtildes0_all_values))
		# print Rtildes0_all_values
		Rtildes0 = Rtildes0_all_values[0]

		for i in range(len(Rtildes0_all_values)):
			if Rtildes0_all_values[i]<Rtildes0:			# Condo is always taking smallest value i.e. gas
				Rtildes0 = Rtildes0_all_values[i]
			else:
				pass

		print 'Rtildes0_all_values are:', Rtildes0_all_values, 'however chosen Rtildes0 is:', Rtildes0

	print 'Is Rtildep0 complex:',Rtildep0
	print 'Is Rtildes0 complex:',Rtildes0

	Rtildep0=abs(Rtildep0)
	Rtildes0=abs(Rtildes0)

	#CHECKING IF PURE VOLUME FRACTION RESULTS ARE VALID.
	# checkVolumeFraction(phip0,'phi_p')
	# checkVolumeFraction(phis0,'phi_s')
	
	#PRINTING OF RESULTS OF PURE FLUID CALCULATIONS.
	#FOR DIAGNOSTIC PURPOSES.
	if verbose:
		print('Rtildep0 = {}, Rtildes0 = {};'.format(Rtildep0,Rtildes0))

	#CALCULATION OF BINARY MIXTURE COMPOSITION AT P, T.
	#default [0.75,0.05]
	#Other good range [0.85,0.05]
		
	Try_all_solvent_guesses = True
	if Try_all_solvent_guesses == False:
		Rtildem,phis = nsolve([EOS_m,(mu_s_m-mu_s0.subs(Rtilde_,Rtildes0))],[Rtilde_,phi_s],[0.85,0.05],verify=True)#, prec=7)
	elif Try_all_solvent_guesses == True:

		Rtildem = 0.0
		phis = 0.0

		Rtildem_all_values = [0.0]
		phis_all_values = [0.0]

		# Rtildem_all_values.append(Rtildem)
		# phis_all_values.append(phis)

		# guess_Rtildem = npy.array([0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99,0.999,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.01,0.02,0.05,0.9999,0.99999,0.999999,0.000001,0.00001,0.0001,0.001])
		# guess_phis = npy.array([0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.0001,0.001,0.000001,0.00001,0.90,0.95,0.97,0.98,0.99,0.999,0.9999,0.99999,0.999999])
		# guess_Rtildem = npy.array([0.50,0.65,0.75,0.85,0.90,0.97,0.99,0.999,0.10,0.20,0.30,0.40,0.01,0.02,0.05,0.9999,0.0001,0.001])
		# guess_phis = npy.array([0.001,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.65,0.75,0.85,0.0001,0.00001,0.90,0.97,0.99,0.999,0.9999])
		# guess_Rtildem = npy.array([0.50,0.70,0.85,0.95,0.999,0.10,0.30,0.01,0.9999,0.001])
		# guess_phis = npy.array([0.001,0.015,0.05,0.10,0.30,0.50,0.80,0.0001,0.00001,0.95,0.999])
		# guess_Rtildem = npy.array([0.50,0.70,0.85,0.95,0.30])
		# guess_phis = npy.array([0.001,0.01,0.10,0.30,0.50,0.80])
		guess_Rtildem = npy.array([0.40,0.60,0.80,0.90])
		guess_phis = npy.array([0.01,0.10,0.30,0.50])
		# guess_Rtildem = npy.array([0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99,0.999,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.01,0.02,0.05,0.9999,0.99999,0.999999,0.000001,0.00001,0.0001,0.001])
		# guess_phis = npy.array([0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.0001,0.001,0.000001,0.00001,0.90,0.95,0.97,0.98,0.99,0.999,0.9999,0.99999,0.999999])

		success = 0
		for i in range(len(guess_Rtildem)):
			for j in range(len(guess_phis)):
				# print 'for loop of Rtildem and phis at i = ', i, 'and j = ', j 
				# print 'for loop of Rtildem and phis at guess_Rtildem[i] = ', guess_Rtildem[i], 'and guess_phis[j] = ', guess_phis[j]

				# print 'line number is:',get_linenumber()
				try:
					Rtildem,phis = nsolve([EOS_m,(mu_s_m-mu_s0.subs(Rtilde_,Rtildes0))],[Rtilde_,phi_s],[guess_Rtildem[i],guess_phis[j]],verify=True)#, prec=7)
					# print 'line number is:',get_linenumber()
				except:
					pass
				Rtildem = complex(Rtildem)
				phis = complex(phis)

				# print 'Is Rtildem complex:',Rtildem
				# print 'Is phis complex:',phis
				
				if Rtildem.real>0.0 and abs(Rtildem.imag)<=10E-3 and phis.real>0.0 and abs(phis.imag)<=10E-15:
					#print 'line number is:',get_linenumber()
					# print 'Is Rtildem complex:',Rtildem
					# print 'Is phis complex:',phis
					Rtildem = abs(Rtildem)
					# Rtildem = round(Rtildem, 6)
					phis = abs(phis)
					# phis = round(phis, 6)
					# print 'Hurry! Rtildem is:', Rtildem, 'and phis is:', phis
					Rtildem_all_values.append(Rtildem)
					phis_all_values.append(phis)
					# print 'Rtildem_all_values is:', Rtildem_all_values
					# print 'phis_all_values is:', phis_all_values
					success = 1
					# break
			# if success == 1:
			# 	break

		# print 'Rtildem_all_values is:', Rtildem_all_values
		# print 'phis_all_values is:', phis_all_values

		# Rtildem_all_values = npy.array(remove_duplicates(Rtildem_all_values))
		# phis_all_values = npy.array(remove_duplicates(phis_all_values))

		Rtildem_all_values,phis_all_values = remove_two_lists_simultaneous_duplicates(Rtildem_all_values,phis_all_values)

		print 'Rtildem_all_values is:', Rtildem_all_values
		print 'phis_all_values is:', phis_all_values
		print 'kier_converted_Condo_theory_phis_all_values is:', phis_all_values*Rtildem_all_values

		# print 'line number is:',get_linenumber()

		Rtildem = Rtildem_all_values[0]
		phis = phis_all_values[0]

		for i in range(len(Rtildem_all_values)):
			if Rtildem_all_values[i]>Rtildem:
				Rtildem = Rtildem_all_values[i]
				phis = phis_all_values[i]
			else:
				pass		

	print 'However chosen value of Rtildem is:', Rtildem
	print 'However chosen value of phis is:', phis
	print 'However chosen value of kier_converted_Condo_theory_phis_all_values is:', phis*Rtildem

	Rtildem=abs(Rtildem)
	phis=abs(phis)

	#CHECKING IF MIXTURE VOLUME FRACTION RESULTS ARE VALID.
	# checkVolumeFraction([Rsm,phis],['Rsm','phi_s'])
	
	#PRINTING OF RESULTS OF MIXTURE COMPOSITION CALCULATIONS.
	#FOR DIAGNOSTIC PURPOSES.
	if verbose:
		print('Rtildem = {}, phis = {};'.format(Rtildem,phis))
	# phip0=0.0 #junk

	return [P,T,Rtildem,phis,Rtildep0,Rtildes0]

def binarySolubilitySwellingCondo_Original(P,T,Mp,Ms,**kwargs):
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

	#Boolean determining whether information is printed.
	#Default option is False.
	verbose = kwargs.get('verbose',False)
	
	# Boolean that determines method of calculation.
	#	 True: Uses simplified (original) swelling calculation assuming pure polymer.
	#	 False: Uses more sophisticated swelling calculation assuming air (N2) content.
	simplified = kwargs.pop('simplified',True)
	
	# CALCULATION OF VOLUME FRACTIONS AT P, T.

	[P,T,Rtildem,cphis,Rtildep0,Rtildes0] = binaryPhaseEquilibriumCondo_Original_nsolve(P,T,Mp,Ms,**kwargs)
	cphip = 1-cphis
	#PURE FLUID PARAMETERS.
	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar
	alpha_p0 = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_s0 = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	
	kphis=cphis*Rtildem
	kphip=cphip*Rtildem
	kphip0=Rtildep0
	kphis0=Rtildes0


	#CALCULATION OF SOLVENT SOLUBILITY (MASS FRACTION) AT P, T.
	ms = (Ms*cphis/alpha_s0)/(Mp*cphip/alpha_p0+Ms*cphis/alpha_s0)
	#CALCULATION OF VOLUME SWELLING (Vp/Vm) IN THE MIXTURE AT P, T.
	Sw = kphip0/(kphip)

	#PRINTING OF RESULTS OF SOLUBILITY AND SWELLING.
	#FOR DIAGNOSTIC PURPOSES.
	if verbose:
		print('ms = {}, Sw = {};'.format(ms,Sw))
	
	return [P,T,ms,Sw]

def calculateBinarySolubilitySwellingCondo_Original(P0,T0,Mp,Ms,**kwargs):

	if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
		XSw = binarySolubilitySwellingCondo_Original(P0,T0,Mp,Ms,**kwargs)
		result = XSw
	
	elif not isListOrNpyArray(T0) and isListOrNpyArray(P0):
		result = [[range(0,len(P0))] for x in range(4)]
		T = range(0,len(P0))
		m_s = range(0,len(P0))
		Sw = range(0,len(P0))
		for i in range(0,len(P0)):
			XSw = binarySolubilitySwellingCondo_Original(P0[i],T0,Mp,Ms,**kwargs)
			T[i] = XSw[1]
			m_s[i] = XSw[2]
			Sw[i] = XSw[3]
		result[0] = P0
		result[1] = T
		result[2] = m_s
		result[3] = Sw

	elif not isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(4)]
		P = range(0,len(T0))
		m_s = range(0,len(T0))
		Sw = range(0,len(T0))
		for i in range(0,len(T0)):
			XSw = binarySolubilitySwellingCondo_Original(P0,T0[i],Mp,Ms,**kwargs)
			P[i] = XSw[0]
			m_s[i] = XSw[2]
			Sw[i] = XSw[3]
		result[0] = P
		result[1] = T0
		result[2] = m_s
		result[3] = Sw
	
	elif isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(4)]
		P = range(0,len(T0))
		m_s = range(0,len(T0))
		Sw = range(0,len(T0))
		for i in range(0,len(T0)):
			XSw = binarySolubilitySwellingCondo_Original(P0[i],T0[i],Mp,Ms,**kwargs)
			P[i] = XSw[0]
			m_s[i] = XSw[2]
			Sw[i] = XSw[3]
		result[0] = P0
		result[1] = T0
		result[2] = m_s
		result[3] = Sw
	
	else:
		raise ValueError('In calculateBinarySwelling: Unknown error involving P0 and T0.')
	
	return result

def calculateBinarySolubilityCondo_Original(P0,T0,Mp,Ms,**kwargs):

	if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
		P,T,m_s,Sw = binarySolubilitySwellingCondo_Original(P0,T0,Mp,Ms,**kwargs)
		result = [P,T,m_s]
	
	elif not isListOrNpyArray(T0) and isListOrNpyArray(P0):
		result = [[range(0,len(P0))] for x in range(3)]
		T = range(0,len(P0))
		m_s = range(0,len(P0))
		for i in range(0,len(P0)):
			X = binarySolubilitySwellingCondo_Original(P0[i],T0,Mp,Ms,**kwargs)
			T[i] = X[1]
			m_s[i] = X[2]
		result[0] = P0
		result[1] = T
		result[2] = m_s

	elif not isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(3)]
		P = range(0,len(T0))
		m_s = range(0,len(T0))
		for i in range(0,len(T0)):
			X = binarySolubilitySwellingCondo_Original(P0,T0[i],Mp,Ms,**kwargs)
			P[i] = X[0]
			m_s[i] = X[2]
		result[0] = P
		result[1] = T0
		result[2] = m_s
	
	elif isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(3)]
		P = range(0,len(T0))
		m_s = range(0,len(T0))
		for i in range(0,len(T0)):
			X = binarySolubilitySwellingCondo_Original(P0[i],T0[i],Mp,Ms,**kwargs)
			m_s[i] = X[2]
		result[0] = P0
		result[1] = T0
		result[2] = m_s
	
	else:
		raise ValueError('In calculateBinarySwelling: Unknown error involving P0 and T0.')
	
	return result

def calculateBinarySwellingCondo_Original(P0,T0,Mp,Ms,**kwargs):

	if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
		P,T,m_s,Sw = binarySolubilitySwellingCondo_Original(P0,T0,Mp,Ms,**kwargs)
		result = [P,T,Sw]

	elif not isListOrNpyArray(T0) and isListOrNpyArray(P0):
		result = [[range(0,len(P0))] for x in range(3)]
		T = range(0,len(P0))
		Sw = range(0,len(P0))
		for i in range(0,len(P0)):
			Swlng = binarySolubilitySwellingCondo_Original(P0[i],T0,Mp,Ms,**kwargs)
			T[i] = Swlng[1]
			Sw[i] = Swlng[3]
		result[0] = P0
		result[1] = T
		result[2] = Sw
	
	elif not isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(3)]
		P = range(0,len(T0))
		Sw = range(0,len(T0))
		for i in range(0,len(T0)):
			Swlng = binarySolubilitySwellingCondo_Original(P0,T0[i],Mp,Ms,**kwargs)
			P[i] = Swlng[0]
			Sw[i] = Swlng[3]
		result[0] = P
		result[1] = T0
		result[2] = Sw
	
	elif isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(3)]
		P = range(0,len(T0))
		Sw = range(0,len(T0))
		for i in range(0,len(T0)):
			Swlng = binarySolubilitySwellingCondo_Original(P0[i],T0[i],Mp,Ms,**kwargs)
			Sw[i] = Swlng[3]
		result[0] = P0
		result[1] = T0
		result[2] = Sw
	
	else:
		raise ValueError('In calculateBinarySwelling: Unknown error involving P0 and T0.')
	
	return result


if __name__ == "__main__":

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
	Hassan=False  
	Hassan_Var_Vol=False  
	Condo=False  
	Condo_Original=True 

	kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight,'Kier':Kier,'Hassan':Hassan,'Hassan_Var_Vol':Hassan_Var_Vol,'Condo':Condo,'Condo_Original':True}

	cepsilon_s,cepsilon_p,cz,czeta,epsilon_p,g,x,delta,zeta=Parameters_for_Mixtures_and_Tg(**kwargs)
	cdelta=100.0


	# P0_X_T4 = npy.concatenate(([0.0001],P0_X_T4),axis=0)
	# T0_X_T4 = npy.concatenate(([373.15],T0_X_T4),axis=0)
	# X0_X_T4 = npy.concatenate(([0],X0_X_T4),axis=0)

	# P0_S_T4 = npy.concatenate(([0.0001],P0_S_T4),axis=0)
	# T0_S_T4 = npy.concatenate(([373.15],T0_S_T4),axis=0)
	# S0_S_T4 = npy.concatenate(([1.0],S0_S_T4),axis=0)

	P0 = npy.linspace(min(P0_X),max(P0_X),3)
	# P0 = npy.linspace(2.0,17,15)
	T1=T0_X_T1[0]	#403	#290
	T2=T0_X_T2[0]	#423	#304
	T3=T0_X_T3[0]	#463	#350
	T4=T0_X_T4[0]	#423	#304
	T5=0.0#T0_X_T5[0]	#463	#350
	T6=0.0#T0_X_T6[0]	#423	#304
	T7=0.0#T0_X_T7[0]	#463	#350
	T8=0.0#T0_X_T8[0]	#463	#350
	T9=0.0#T0_X_T9[0]	#463	#350

	number_of_points = 3
	P1 = npy.linspace(min(P0_X),max(P0_X_T1),number_of_points)
	P2 = npy.linspace(min(P0_X),max(P0_X_T2),number_of_points)
	P3 = npy.linspace(min(P0_X),max(P0_X_T3),number_of_points)
	P4 = npy.linspace(min(P0_X),max(P0_X_T4),number_of_points)
	# P5 = npy.linspace(min(P0_X),max(P0_X_T5),number_of_points)

	result = calculateBinarySolubilitySwellingCondo_Original(P1,T1,Mp,Ms,zeta=zeta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,verbose=True)
	Xs_T1_DHV = result[2]
	Sw_T1_DHV = result[3]

	print Xs_T1_DHV, Sw_T1_DHV