
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
# from calculateBinaryVariablesCHV import *
from Parameters_of_Different_Polymers import *
from Parameters_for_Mixtures_and_Tg import *
from All_Functions import calculateThermodynamicVariables, EOS_pure_k
from SplittingExperimentalDataOf_X_Sw_aboveANDbelowTgANDIsotherms import *
# from Tait_Parameters_of_Different_Polymers import *
# from loadExperimentalDataCO2 import *
# from CO2PVT_interpolation import *
from scipy.optimize import *
from Split_Exp_Data_in_Isotherms import*
from collections import OrderedDict			#For Exotic Line Styles
import cmath
from sympy import *
import types
from inspect import currentframe #To get line number in Print
# from self_bisect import *
from To_get_colored_print import *

def self_bisect_Tg(T1,T2,P,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward):

	# prPurple('number_of_trails={}'.format(number_of_trails))
	criterion1 = Find_Tg_Bisect_xS_infty(T1,P,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
	criterion2 = Find_Tg_Bisect_xS_infty(T2,P,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)

	if (criterion1<0 and criterion2<0) or (criterion1>0 and criterion2>0):
		Tg = 0.0
		prGreen('Failed! Sign of both criterions are the same at T1={} and T2= {}'.format(T1,T2))

	if (criterion1>0 and criterion2<0) or (criterion1<0 and criterion2>0):
		prRed('Hurry! Different Sign found at T1={} and T2= {}'.format(T1,T2))
		Tg = bisect(Find_Tg_Bisect_xS_infty,T1,T2,args=(P,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward),xtol=1E-2)
		# Tg = brentq(Find_Tg_Bisect_xS_infty,T1,T2,args=(P,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward),xtol=1E-3)

	return Tg

def self_bisect_Pg(P1,P2,T,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward):

	# prPurple('number_of_trails={}'.format(number_of_trails))
	criterion1 = Find_Pg_Bisect_xS_infty(P1,T,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
	criterion2 = Find_Pg_Bisect_xS_infty(P2,T,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)

	if (criterion1<0 and criterion2<0) or (criterion1>0 and criterion2>0):
		Pg = 0.0
		prGreen('Failed! Sign of both criterions are the same at P1={} and P2= {}'.format(P1,P2))

	if (criterion1>0 and criterion2<0) or (criterion1<0 and criterion2>0):
		prRed('Hurry! Different Sign found at P1={} and P2= {}'.format(P1,P2))
		Pg = bisect(Find_Pg_Bisect_xS_infty,P1,P2,args=(T,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward),xtol=1E-2)
		# Pg = brentq(Find_Pg_Bisect_xS_infty,P1,P2,args=(T,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward),xtol=1E-3)

	return Pg

def get_linenumber():
	#To get line number in Print
    cf = currentframe()
    return cf.f_back.f_lineno

def discard_zeros(x,y):
	
	for i in range(len(x)):
		if x[i]==0:
			y[i]=0

	for i in range(len(y)):
		if y[i]==0:
			x[i]=0

	x = npy.delete(x, npy.argwhere( (x >= 0) & (x <= 0) ))
	y = npy.delete(y, npy.argwhere( (y >= 0) & (y <= 0) ))

	return x,y

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

def binaryPhaseEquilibriumCHV(P,T,Mp,Ms,**kwargs):
	# print 'first line of binaryPhaseEquilibriumCHV'
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

	# guess1 = npy.array([0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99,0.999,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.01,0.02,0.05,0.9999,0.99999,0.999999,0.000001,0.00001,0.0001,0.001])
	# guess1 = npy.array([0.50,0.65,0.75,0.85,0.90,0.97,0.99,0.999,0.10,0.20,0.30,0.40,0.01,0.02,0.05,0.9999,0.0001,0.001])
	guess1 = npy.array([0.99,0.999,0.95,0.85,0.7,0.5,0.30,0.10,0.01,0.05])

	for i in range(len(guess1)):
		# print 'for loop of phip0:', i
		try:
			phip0 = nsolve(EOS_p0,phi_p,guess1[i],verify=True)
		except:
			pass
		
		phip0 = complex(phip0)

		if phip0.real>0.0 and abs(phip0.imag)<=10E-3:
			# print 'Is phip0 complex:',phip0
			phip0 = abs(phip0)
			# phip0 = round(phip0, 6)
			# print Ppstar, Tpstar, Rpstar, Mp, P, T, phip0
			residual = EOS_pure_k(phip0,P,T,Mp,Ppstar,Tpstar,Rpstar)
			# print 'Hurry! phip0 is:', phip0, 'and residual is:', residual
			phip0_all_values.append(phip0)
			# break				#Do not break it because it is detecting phi-->0 as a possible solution. But, phi---> is correct solution
		else:
			# phip0 = 0.0
			pass

	phip0_all_values = npy.array(remove_duplicates(phip0_all_values))
	# print phip0_all_values
	phip0 = phip0_all_values[0]

	for i in range(len(phip0_all_values)):
		if phip0_all_values[i]>phip0:
			phip0 = phip0_all_values[i]
		else:
			pass	

	print 'phip0_all_values are:', phip0_all_values, 'however chosen phip0 is:', phip0

	if phip0==0.0:
		print 'Program Failed to get value of phip0'
		# raise ValueError('Program Failed to get value of phip0')

	phis0 = 0.0
	phis0_all_values = []

	# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.000001,0.00001,0.90,0.95,0.97,0.98,0.99,0.999,0.9999,0.99999,0.999999])
	# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.65,0.75,0.85,0.00001,0.90,0.97,0.99,0.999,0.9999])
	# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.30,0.45,0.70,0.90,0.95,0.999])
	guess2 = npy.array([0.65,0.85,0.95,0.45,0.30,0.01,0.10])

	for i in range(len(guess2)):
		# print 'for loop of phis0:', i

		try:
			phis0 = nsolve(EOS_s0,phi_s,guess2[i],verify=True)
		except:
			pass

		phis0 = complex(phis0)

		if phis0.real>0.0 and abs(phis0.imag)<=10E-3:
			# print 'Is phis0 complex:',phis0
			phis0 = abs(phis0)
			# phis0 = round(phis0, 6)
			residual = EOS_pure_k(phis0,P,T,Ms,Psstar,Tsstar,Rsstar)
			# residual_high = EOS_pure_k(phis0,P,T+1,Ms,Psstar,Tsstar,Rsstar)
			# residual_low = EOS_pure_k(phis0,P,T-1,Ms,Psstar,Tsstar,Rsstar)
			# print residual_high<0, residual_low<0
			# print Psstar, Tsstar, Rsstar, Ms, P, T, phis0
			# print 'Hurry! phis0 is:', phis0, 'and residual is:', residual
			phis0_all_values.append(phis0)
			# break			#Do not break it because CO2 has multiple solution.
		else:
			# phis0 = 0.0
			pass

	phis0_all_values = npy.array(remove_duplicates(phis0_all_values))
	# print phis0_all_values
	phis0 = phis0_all_values[0]

	for i in range(len(phis0_all_values)):
		if phis0_all_values[i]>phis0:
			phis0 = phis0_all_values[i]
		else:
			pass

	print 'phis0_all_values are:', phis0_all_values, 'however chosen phis0 is:', phis0

	if phis0==0.0:
		print 'Program Failed to get value of phis0'
		# raise ValueError('Program Failed to get value of phis0')

	# print 'Is phip0 complex:',phip0
	# print 'Is phis0 complex:',phis0

	# phip0=abs(phip0)
	# phis0=abs(phis0)

	#CHECKING IF PURE VOLUME FRACTION RESULTS ARE VALID.
	checkVolumeFraction(phip0,'phi_p')
	checkVolumeFraction(phis0,'phi_s')
	
	#PRINTING OF RESULTS OF PURE FLUID CALCULATIONS.
	#FOR DIAGNOSTIC PURPOSES.
	# if verbose:
	print('phip0 = {}, phis0 = {};'.format(phip0,phis0))
	
	# mu_gasPhase=mu_s0.subs(phi_s,phis0)
	# print mu_gasPhase

	#CALCULATION OF BINARY MIXTURE COMPOSITION AT P, T.
	#default [0.75,0.05]
	#Other good range [0.85,0.10]

	phip = 0.0
	phis = 0.0

	phip_all_values = [0.0]
	phis_all_values = [0.0]

	# phip_all_values.append(phip)
	# phis_all_values.append(phis)

	# guess_phip = npy.array([0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99,0.999,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.01,0.02,0.05,0.9999,0.99999,0.999999,0.000001,0.00001,0.0001,0.001])
	# guess_phis = npy.array([0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.0001,0.001,0.000001,0.00001,0.90,0.95,0.97,0.98,0.99,0.999,0.9999,0.99999,0.999999])
	# guess_phip = npy.array([0.50,0.65,0.75,0.85,0.90,0.97,0.99,0.999,0.10,0.20,0.30,0.40,0.01,0.02,0.05,0.9999,0.0001,0.001])
	# guess_phis = npy.array([0.001,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.65,0.75,0.85,0.0001,0.00001,0.90,0.97,0.99,0.999,0.9999])
	# guess_phip = npy.array([0.50,0.70,0.85,0.95,0.999,0.10,0.30,0.01,0.9999,0.001])
	# guess_phis = npy.array([0.001,0.015,0.05,0.10,0.30,0.50,0.80,0.0001,0.00001,0.95,0.999])
	# guess_phip = npy.array([0.50,0.70,0.85,0.95,0.30])
	# guess_phis = npy.array([0.001,0.01,0.10,0.30,0.50,0.80])
	guess_phip = npy.array([0.40,0.60,0.80,0.90])
	guess_phis = npy.array([0.01,0.10,0.30,0.50])

	for i in range(len(guess_phip)):
		for j in range(len(guess_phis)):
			# print 'for loop of phip and phis at i = ', i, 'and j = ', j 
			# print 'line number is:',get_linenumber()
			try:
				phip,phis = nsolve([EOS_m,(mu_s_m-mu_s0.subs(phi_s,phis0))],[phi_p,phi_s],[guess_phip[i],guess_phis[j]],verify=True)
				# print 'line number is:',get_linenumber()
			except:
				pass
			phip = complex(phip)
			phis = complex(phis)

			# print 'Is phip complex:',phip
			# print 'Is phis complex:',phis
			
			if phip.real>0.0 and abs(phip.imag)<=10E-3 and phis.real>0.0 and abs(phis.imag)<=10E-15:
				#print 'line number is:',get_linenumber()
				# print 'Is phip complex:',phip
				# print 'Is phis complex:',phis
				phip = abs(phip)
				# phip = round(phip, 6)
				phis = abs(phis)
				# phis = round(phis, 6)
				# print 'Hurry! phip is:', phip, 'and phis is:', phis
				phip_all_values.append(phip)
				phis_all_values.append(phis)
				# print 'phip_all_values is:', phip_all_values
				# print 'phis_all_values is:', phis_all_values
				# break
		# break

	# print 'phip_all_values is:', phip_all_values
	# print 'phis_all_values is:', phis_all_values

	# phip_all_values = npy.array(remove_duplicates(phip_all_values))
	# phis_all_values = npy.array(remove_duplicates(phis_all_values))

	phip_all_values,phis_all_values = remove_two_lists_simultaneous_duplicates(phip_all_values,phis_all_values)

	print 'phip_all_values is:', phip_all_values
	print 'phis_all_values is:', phis_all_values

	print 'line number is:',get_linenumber()

	phip = phip_all_values[0]
	phis = phis_all_values[0]

	for i in range(len(phip_all_values)):
		if phip_all_values[i]>phip:
			phip = phip_all_values[i]
			phis = phis_all_values[i]
		else:
			pass		

	#print 'line number is:',get_linenumber()
	if phip==0.0 or phis==0.0:
		print 'Program Failed to get value of phip and phis'
		# raise ValueError('Program Failed to get value of phip and phis')

	# phip,phis = nsolve([EOS_m,(mu_s_m-mu_s0.subs(phi_s,phis0))],[phi_p,phi_s],[0.85,0.05],verify=True)

	# print phip_all_values
	# print phis_all_values
	#print 'line number is:',get_linenumber()
	#print 'line number is:',get_linenumber()

	# print 'Is phip complex:', phip
	# print 'Is phis complex:', phis
	
	# phip=abs(phip)
	# phis=abs(phis)

	#CHECKING IF MIXTURE VOLUME FRACTION RESULTS ARE VALID.
	checkVolumeFraction([phip,phis],['phi_p','phi_s'])
	
	#PRINTING OF RESULTS OF MIXTURE COMPOSITION CALCULATIONS.
	#FOR DIAGNOSTIC PURPOSES.
	# if verbose:
	print('returning: phip = {}, phis = {};'.format(phip,phis))
	# phip0=0.0 #junk
	# print 'last line of binaryPhaseEquilibriumCHV'

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

def Find_Tg_Bisect_xS_infty(Tg,P,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward):

	result = calculateBinarySolubilitySwelling('CHV',P,Tg,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol,forward=forward,backward=backward)
	Xs = result[2]
	Sw = result[3]
	phip = result[4]
	phis = result[5]
	Rtilde = result[6]
	phip0 = result[7]
	phis0 = result[8]

	properties=calculateThermodynamicVariables(P,Tg,phip,phis,phip0,phis0,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
	print properties
	S_1 = properties[0]
	S_2 = properties[1]

	criterion=S_1-x*S_infty
	print 'S_1:', S_1,' x*S_infty:', x*S_infty
	# print 'P=',P,'T=',Tg,'phip=',phip,'phis=',phis,'phip0=',phip0,'phis0=',phis0,'Mp=',Mp,'Ms=',Ms,'g=',g,'epsilon_p=',epsilon_p,'zeta=',zeta,'delta=',delta,'Ppstar=',Ppstar,'Tpstar=',Tpstar,'Rpstar=',Rpstar,'Psstar=',Psstar,'Tsstar=',Tsstar,'Rsstar=',Rsstar

	return criterion

def Find_Pg_Bisect_xS_infty(Pg,T,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward):

	result = calculateBinarySolubilitySwelling('CHV',Pg,T,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol,forward=forward,backward=backward)
	Xs = result[2]
	Sw = result[3]
	phip = result[4]
	phis = result[5]
	Rtilde = result[6]
	phip0 = result[7]
	phis0 = result[8]

	properties=calculateThermodynamicVariables(Pg,T,phip,phis,phip0,phis0,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
	S_1 = properties[0]
	S_2 = properties[1]
	print 'S_1:', S_1,' x*S_infty:', x*S_infty
	# print 'P=',P,'T=',Tg,'phip=',phip,'phis=',phis,'phip0=',phip0,'phis0=',phis0,'Mp=',Mp,'Ms=',Ms,'g=',g,'epsilon_p=',epsilon_p,'zeta=',zeta,'delta=',delta,'Ppstar=',Ppstar,'Tpstar=',Tpstar,'Rpstar=',Rpstar,'Psstar=',Psstar,'Tsstar=',Tsstar,'Rsstar=',Rsstar
	
	criterion=S_1-x*S_infty

	return criterion

def GlassTemperature(direction,P,Mp,Ms,**kwargs):
	
	for key,value in kwargs.items():
		exec "%s=%s" % (key,value)

	min_Tg=250.0
	max_Tg=360.0
	# step_Tg=10
	num_of_points = 11

	if direction=='fwd':
		start=min_Tg
		end=max_Tg
		# step=step_Tg
		# print 'forward'
		
	elif direction=='bwd':
		start=max_Tg
		end=min_Tg
		# step=-1*step_Tg
		# print 'backward'

	T_array = npy.linspace(start, end, num=num_of_points)

	for i in range(len(T_array)-1):
		Tg=0.0
		T1 = T_array[i]
		T2 = T_array[i+1]
		# Tg= self_bisect_Tg(T1,T2,P,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
		try:
			Tg= self_bisect_Tg(T1,T2,P,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
		except:
			pass

		if Tg!=0.0:
			prRed('Hurry! Tg is:{} for direction {}'.format(Tg,direction))
			break
	if Tg==0.0:
		print 'Program Failed to get value of Tg in given bisect range in direction', direction

	return Tg

def GlassPressure(direction,T,Mp,Ms,**kwargs):
	
	for key,value in kwargs.items():
		exec "%s=%s" % (key,value)

	min_Pg=0.101325
	max_Pg=15.0
	# step_Pg=1
	num_of_points = 11

	if direction=='fwd':
		start=min_Pg
		end=max_Pg
		# step=step_Pg
		# print 'forward'
		
	elif direction=='bwd':
		start=max_Pg
		end=min_Pg
		# step=-1*step_Pg
		# print 'backward'

	P_array = npy.linspace(start, end, num=num_of_points)

	for i in range(len(P_array)-1):
		Pg=0.0
		P1 = P_array[i]
		P2 = P_array[i+1]
		# Pg = self_bisect_Pg(P1,P2,T,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)

		try:
			Pg = self_bisect_Pg(P1,P2,T,S_infty,Mp,Ms,g,epsilon_p,x,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
		except:
			pass

		if Pg!=0.0:
			prRed('Hurry! Pg is:{} for direction {}'.format(Pg,direction))
			break
	if Pg==0.0:
		print 'Program Failed to get value of Pg in given bisect range in direction', direction

	return Pg

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
Hassan=True  
Hassan_Var_Vol=False  
Condo=False  
Condo_Original=False 

kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight,'Kier':Kier,'Hassan':Hassan,'Hassan_Var_Vol':Hassan_Var_Vol,'Condo':Condo,'Condo_Original':True}

cepsilon_s,cepsilon_p,cz,czeta,epsilon_p,g,x,delta,zeta=Parameters_for_Mixtures_and_Tg(**kwargs)
cdelta=100.0

xS_infty=x*(Ppstar/(Tpstar*Rpstar))*(1+ln(1+g))
S_infty=(Ppstar/(Tpstar*Rpstar))*(1+ln(1+g))
# print Ppstar,Tpstar,Rpstar,g,epsilon_p,x,xS_infty

forward=False		#Not 100% sure: Do not use forward=True and backward=False because: If forward=True and backward=False, then backward=False is penerating deep into the code and causing forward=True to not give any answers. i.e. all values are failing. 
backward=True

Find_Tg_at_P=True
P=npy.linspace(0.101325,15.0,20)

Find_Pg_at_T=False
T=npy.linspace(250,360,15)

zeta= 0.88
# delta=1.00
print 'alpha_pure_s =', (Psstar*Ms)/(kB*Tsstar*Rsstar)
print 'alpha_pure_p =', (Ppstar*Mp)/(kB*Tpstar*Rpstar)

if Find_Tg_at_P:
	#For Kier or Hassan or Condo:
	Tg_bisect_fwd=npy.zeros(len(P))
	Tg_bisect_bwd=npy.zeros(len(P))

if Find_Pg_at_T:
	#For Kier or Hassan or Condo:
	Pg_bisect_fwd=npy.zeros(len(T))
	Pg_bisect_bwd=npy.zeros(len(T))

if Kier or Hassan or Condo or Hassan_Var_Vol:

	kwargs = {'S_infty':S_infty,'g':g,'epsilon_p':epsilon_p,'x':x,'zeta':zeta,'delta':delta,'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar,'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar,'Kier':Kier,'Hassan':Hassan,'Hassan_Var_Vol':Hassan_Var_Vol,'Condo':Condo,'forward':forward,'backward':backward}

	if Find_Tg_at_P:
		for i in range(0,len(P)):
			print 'Iterating for P:', P[i], 'for bisect method'
			if forward:
				Tg_bisect_fwd[i] = GlassTemperature('fwd',P[i],Mp,Ms,**kwargs)
			if backward:
				Tg_bisect_bwd[i] = GlassTemperature('bwd',P[i],Mp,Ms,**kwargs)

	if Find_Pg_at_T:
		for i in range(0,len(T)):
			print 'Iterating for T:', T[i], 'for bisect method'
			if forward:
				Pg_bisect_fwd[i] = GlassPressure('fwd',T[i],Mp,Ms,**kwargs)
			if backward:
				Pg_bisect_bwd[i] = GlassPressure('bwd',T[i],Mp,Ms,**kwargs)

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

if Kier or Hassan or Condo or Hassan_Var_Vol:
	if Find_Tg_at_P:
		if forward:
			print 'P before deleting zeros = ', npy.array(P)
			print 'Tg_bisect_fwd before deleting zeros = ', npy.array(Tg_bisect_fwd)
			P_fwd,Tg_bisect_fwd=discard_zeros(P,Tg_bisect_fwd)
			print 'P = ', npy.array(P_fwd)
			print 'Tg_bisect_fwd = ', npy.array(Tg_bisect_fwd)
		if backward:
			print 'P before deleting zeros = ', npy.array(P)
			print 'Tg_bisect_bwd before deleting zeros = ', npy.array(Tg_bisect_bwd)
			P_bwd,Tg_bisect_bwd=discard_zeros(P,Tg_bisect_bwd)
			print 'P = ', npy.array(P_bwd)
			print 'Tg_bisect_bwd = ', npy.array(Tg_bisect_bwd)

	if Find_Pg_at_T:
		if forward:
			print 'Pg_bisect_fwd before deleting zeros = ', npy.array(Pg_bisect_fwd)
			print 'T  before deleting zeros = ', npy.array(T)
			Pg_bisect_fwd,T_fwd=discard_zeros(Pg_bisect_fwd,T)
			print 'Pg_bisect_fwd = ', npy.array(Pg_bisect_fwd)
			print 'T = ', npy.array(T_fwd)
		if backward:
			print 'Pg_bisect_bwd before deleting zeros = ', npy.array(Pg_bisect_bwd)
			print 'T before deleting zeros = ', npy.array(T)
			Pg_bisect_bwd,T_bwd=discard_zeros(Pg_bisect_bwd,T)
			print 'Pg_bisect_bwd = ', npy.array(Pg_bisect_bwd)
			print 'T = ', npy.array(T_bwd)

#Setting font size
axis_size = 20
title_size = 20
size = 14
label_size = 20
plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size

#Setting saved image properties
img_extension = '.png'
img_dpi = None
output_folder = 'ChangingzetaofPMMA'

#Checking for existence of output directory. If such a directory doesn't exist, one is created.
if not os.path.exists('./'+output_folder):
    os.makedirs('./'+output_folder)

#General line properties.
linewidth = 1
markersize = 6

arrow_ls = 'dashdot'
show_arrows = True

#==================================================================================
#Plots.
figPUREPS=plt.figure(num=None, figsize=(10,6), dpi=img_dpi, facecolor='w', edgecolor='k')
ax = plt.axes()

# plt.plot(P_exp,Tg_exp,color='b',marker='o',ls='',label='Tg_exp_condo',ms=markersize)
# plt.plot(P_exp_Condo,Tg_exp_Condo,color='k',marker='o',ls='',label='Tg_exp_condo',ms=markersize)

if Kier or Hassan or Condo or Hassan_Var_Vol:
	if Find_Tg_at_P:
		if forward:
			plt.plot(P_fwd,Tg_bisect_fwd,color='r',marker='x',lw=linewidth,ls='-.',label='Tg_bisect_fwd')
		if backward:
			plt.plot(P_bwd,Tg_bisect_bwd,color='b',marker='o',lw=linewidth,ls='-',label='Tg_bisect_bwd')

	if Find_Pg_at_T:
		if forward:
			plt.plot(Pg_bisect_fwd,T_fwd,color='g',marker='v',lw=linewidth,ls='-.',label='Pg_bisect_fwd')
		if backward:
			plt.plot(Pg_bisect_bwd,T_bwd,color='m',marker='s',lw=linewidth,ls='-',label='Pg_bisect_bwd')

plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
plt.ylabel(r'Glass Temperature Tg (K)',fontsize=axis_size)
#plt.axis([300,500,0,1.5])
plt.legend(loc=1,fontsize=size,numpoints=1)
# plt.title(kwargs, fontdict=None, loc='center', pad=None)
plt.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.10,wspace=0.30,hspace=0.25)
figPUREPS.savefig('./'+output_folder+r'\PMMA_CO2_Self_Grassia_02kilo_POST_THESIS_Paper15_Tg(P)'+img_extension,dpi=240)

#Show plot windows.
plt.show()



