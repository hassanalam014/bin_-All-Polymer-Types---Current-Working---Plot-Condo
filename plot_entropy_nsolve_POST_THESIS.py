
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
from find_discontinuity import *

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

	min_Tg=306.75
	max_Tg=307
	# step_Tg=10
	num_of_points = 25

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

	min_Pg=6.5
	max_Pg=7.4
	# step_Pg=1
	num_of_points = 20

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

print Psstar,Tsstar,Rsstar,Ppstar,Tpstar,Rpstar,g,epsilon_p,x,xS_infty,zeta,delta

Isotherms=False
Isobars=True
Entropy=True #Isobars
Plot_Phi=False #Isobars

forward=False		#Not 100% sure: Do not use forward=True and backward=False because: If forward=True and backward=False, then backward=False is penerating deep into the code and causing forward=True to not give any answers. i.e. all values are failing. 
backward=True

Plot_S2=False	

if Isotherms:
	# P0 = npy.linspace(6,23,10)	
	if T0_X!=[]:
		# P0 = npy.linspace(min(P0_X),max(P0_X),7)
		P0 = npy.linspace(3.0,4.0,10)
		T1=300#T0_X_T1[0]	#403	#290
		T2=0.0#T0_X_T2[0]	#423	#304
		T3=0.0#T0_X_T3[0]	#463	#350
		T4=0.0#T0_X_T4[0]	#423	#304
		T5=0.0#T0_X_T5[0]	#463	#350
		T6=0.0#T0_X_T6[0]	#423	#304
		T7=0.0#T0_X_T7[0]	#463	#350
		T8=0.0#T0_X_T8[0]	#463	#350

	if T0_S!=[] and False:
		# P0 = npy.linspace(min(P0_S),max(P0_S),5)
		# P0 = npy.linspace(0.301325,30,20)
		T1=0.0#T0_S_T1[0]	#403	#290
		T2=0.0#T0_S_T2[0]	#423	#304
		T3=0.0#T0_S_T3[0]	#463	#350
		T4=0.0#T0_S_T4[0]	#423	#304
		T5=0.0#T0_S_T5[0]	#463	#350
		T6=0.0#T0_S_T6[0]	#423	#304
		T7=0.0#T0_S_T7[0]	#463	#350
		T8=0.0#T0_S_T8[0]	#463	#350

# zeta = 0.93	#0.90
# czeta = 1.0875
if Isobars:
	number_of_isobar=3
	# T0 = npy.linspace(min(T0_X),max(T0_X),20)
	T0 = npy.linspace(290,330,2)		#max: 1400000  #Small pressure ==> entropy max reaches at smaller temperature
	P1=5.0	#P0_X_P1[0]#0.101325
	P2=7.15	#30.0
	P3=9.0	#50.0


if Isotherms:
	if T1 != 0.0:
		P1_discontinuity = find_discontinuity_pressure(T1,Psstar,Tsstar,Rsstar,Ms)
		# print (math.isnan(P1_discontinuity)), P1_discontinuity
		if not math.isnan(P1_discontinuity):
			P0 = npy.append(P0,[P1_discontinuity-0.001,P1_discontinuity+0.001] )
			P0 = npy.sort(P0)
	if T2 != 0.0:
		P2_discontinuity = find_discontinuity_pressure(T2,Psstar,Tsstar,Rsstar,Ms)
		# print (math.isnan(P2_discontinuity)), P2_discontinuity
		if not math.isnan(P2_discontinuity):
			P0 = npy.append(P0,[P2_discontinuity-0.001,P2_discontinuity+0.001] )
			P0 = npy.sort(P0)
	if T3 != 0.0:
		P3_discontinuity = find_discontinuity_pressure(T3,Psstar,Tsstar,Rsstar,Ms)
		# print (math.isnan(P3_discontinuity)), P3_discontinuity
		if not math.isnan(P3_discontinuity):
			P0 = npy.append(P0,[P3_discontinuity-0.001,P3_discontinuity+0.001] )
			P0 = npy.sort(P0)


if Isobars:
	if P1 != 0.0:
		T1_discontinuity = find_discontinuity_temperature(P1,Psstar,Tsstar,Rsstar,Ms)
		# print (math.isnan(T1_discontinuity)), T1_discontinuity
		if not math.isnan(T1_discontinuity):
			T0 = npy.append(T0,[T1_discontinuity,T1_discontinuity+0.08] )
			T0 = npy.sort(T0)
	if P2 != 0.0:
		T2_discontinuity = find_discontinuity_temperature(P2,Psstar,Tsstar,Rsstar,Ms)
		# print (math.isnan(T2_discontinuity)), T2_discontinuity
		if not math.isnan(T2_discontinuity):
			T0 = npy.append(T0,[T2_discontinuity,T2_discontinuity+0.07] )
			T0 = npy.sort(T0)
	if P3 != 0.0:
		T3_discontinuity = find_discontinuity_temperature(P3,Psstar,Tsstar,Rsstar,Ms)
		# print (math.isnan(T3_discontinuity)), T3_discontinuity
		if not math.isnan(T3_discontinuity):
			T0 = npy.append(T0,[T3_discontinuity,T3_discontinuity+0.07] )
			T0 = npy.sort(T0)


# print 'T0 = ', T0
# print type(T0)
# kajlsgjaslg
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
	if Isotherms:
		isotherm_label=1
		for i in range(0,number_of_isotherm):
			source='''if T%s!=0.0:
				result = calculateBinarySolubilitySwelling('CHV',P0,T%s,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				Xs_T%s_DHV = npy.array(result[2])
				Sw_T%s_DHV = npy.array(result[3])
				phip_T%s_DHV = npy.array(result[4])
				phis_T%s_DHV = npy.array(result[5])
				Rtilde_T%s_DHV = npy.array(result[6])
				phip0_T%s_DHV = npy.array(result[7])
				phis0_T%s_DHV = npy.array(result[8])

				if Entropy:
					properties=calculateThermodynamicVariables(P0,T%s,phip_T%s_DHV,phis_T%s_DHV,phip0_T%s_DHV,phis0_T%s_DHV,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
					S_1_T%s_DHV = npy.array(properties[2])
					S_2_T%s_DHV = npy.array(properties[3])'''
			
			exec source %(isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label)

			isotherm_label=isotherm_label+1

			######################################

	if Isobars:
		isobar_label=1
		for i in range(0,number_of_isobar):
			source='''if P%s!=0.0:
				result = calculateBinarySolubilitySwelling('CHV',P%s,T0,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				Xs_P%s_DHV = result[2]
				Sw_P%s_DHV = result[3]
				phip_P%s_DHV = result[4]
				phis_P%s_DHV = result[5]
				Rtilde_P%s_DHV = result[6]
				phip0_P%s_DHV = result[7]
				phis0_P%s_DHV = result[8]
				
				if Entropy:
					properties=calculateThermodynamicVariables(P%s,T0,phip_P%s_DHV,phis_P%s_DHV,phip0_P%s_DHV,phis0_P%s_DHV,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
					S_1_P%s_DHV = properties[2]
					S_2_P%s_DHV = properties[3]'''
			
			exec source %(isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label)
			isobar_label=isobar_label+1
		
			########################################


#Setting saved image properties
img_extension = '.png'
img_dpi = None
output_folder = 'Post_Thesis'

#Checking for existence of output directory. If such a directory doesn't exist, one is created.
if not os.path.exists('./'+output_folder):
    os.makedirs('./'+output_folder)

#Setting font size
axis_size = 28  #Size  of x and y axis wordings (names)
title_size = 20 #We have no title
size = 18		#Size of legendre
label_size = 26	#Size of values of ticks on x and y axis

plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size

#Defining axes
phi_axes = [5,25,0.0,1.0]
TD_axes = [5,25,0.0,5.0]

#Markers
mark1 = 'o'
mark2 = 's'
mark3 = '>'
mark4 = '<'
mark5 = 'D'
mark6 = 'H'
mark7 = 'P'
mark8 = 'X'

#Linestyles
ls1 = '-'
ls2 = '--'
ls3 = ':'

#General line properties.
linewidth = 4
markersize = 12

if Kier or Hassan or Condo or Hassan_Var_Vol:
	if Isotherms:
		if Plot_Phi:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			if T1!=0.0:
				plt.plot(P0,phip_T1_DHV,'r',ls=ls1,label='phi_p_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phip_T2_DHV,'r',ls=ls2,label='phi_p_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phip_T3_DHV,'r',ls=ls3,label='phi_p_{} K'.format(T3),lw=linewidth)
			
		
			if T1!=0.0:
				plt.plot(P0,phis_T1_DHV,'m',ls=ls1,label='phi_s_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phis_T2_DHV,'m',ls=ls2,label='phi_s_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phis_T3_DHV,'m',ls=ls3,label='phi_s_{} K'.format(T3),lw=linewidth)
			
			
			if T1!=0.0:
				plt.plot(P0,Rtilde_T1_DHV,'b',ls=ls1,label='Rtilde_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Rtilde_T2_DHV,'b',ls=ls2,label='Rtilde_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Rtilde_T3_DHV,'b',ls=ls3,label='Rtilde_{} K'.format(T3),lw=linewidth)
			

			if T1!=0.0:
				plt.plot(P0,phip0_T1_DHV,'k',ls=ls1,label='phi_p0_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phip0_T2_DHV,'k',ls=ls2,label='phi_p0_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phip0_T3_DHV,'k',ls=ls3,label='phi_p0_{} K'.format(T3),lw=linewidth)
			

			if T1!=0.0:
				plt.plot(P0,phis0_T1_DHV,'y',ls=ls1,label='phi_s0_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,phis0_T2_DHV,'y',ls=ls2,label='phi_s0_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,phis0_T3_DHV,'y',ls=ls3,label='phi_s0_{} K'.format(T3),lw=linewidth)

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('phi',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(phi_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Entropy:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			
			if T1!=0.0:
				plt.plot(P0,S_1_T1_DHV,'r',ls=ls2,label='S_1_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,S_1_T2_DHV,'b',ls=ls1,label='S_1_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,S_1_T3_DHV,'g',ls=ls3,label='S_1_{} K'.format(T3),lw=linewidth)

			if Plot_S2:
				if T1!=0.0:
					plt.plot(P0,S_2_T1_DHV,'m',ls=ls1,label='S_2_{} K'.format(T1),lw=linewidth)
				if T2!=0.0:
					plt.plot(P0,S_2_T2_DHV,'m',ls=ls2,label='S_2_{} K'.format(T2),lw=linewidth)
				if T3!=0.0:
					plt.plot(P0,S_2_T3_DHV,'m',ls=ls3,label='S_2_{} K'.format(T3),lw=linewidth)

			# plt.axvline(x=378,lw=0.5,color='k', linestyle='-.')
			plt.axhline(y=0.0,lw=0.5,color='k', linestyle='-.')
			# S_max=npy.max(S_1_P1)
			# print 'S_max is:', S_max
			Tg_line=xS_infty#0.317*0.98268#0.310707*0.8708171#0.2361#2.38*0.271##0.2558 #S_max*kx #0.2361
			plt.axhline(y=Tg_line,lw=0.5,color='k', linestyle='-.')

			plt.xlabel('Pressure P (MPa)',fontsize=axis_size)
			plt.ylabel('Entropy',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(TD_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

	# T0 = [250.0, 251.86440678, 253.72881356, 255.59322034, 257.45762712, 259.3220339, 261.18644068, 263.05084746, 264.91525424, 266.77966102, 268.6440678, 270.50847458, 272.37288136, 274.23728814, 276.10169492, 277.96610169, 279.83050847, 281.69491525, 283.55932203, 284.58374829, 284.66374829, 285.42372881, 287.28813559, 289.15254237, 290.09918095, 290.16918095, 291.01694915, 292.88135593, 294.74576271, 296.35239714, 296.42239714, 296.61016949, 298.47457627, 300.33898305, 302.20338983, 304.06779661, 305.93220339, 307.79661017, 309.66101695, 311.52542373, 313.38983051, 315.25423729, 317.11864407, 318.98305085, 320.84745763, 322.71186441, 324.57627119, 326.44067797, 328.30508475, 330.16949153, 332.03389831, 333.89830508, 335.76271186, 337.62711864, 339.49152542, 341.3559322, 343.22033898, 345.08474576, 346.94915254, 348.81355932, 350.6779661, 352.54237288, 354.40677966, 356.27118644, 358.13559322, 360.0]
	# S_1_P1_DHV =  [0.287361847792063, 0.291977922320856, 0.296529918157662, 0.301007532620532, 0.305400599881289, 0.309699376537802, 0.313894818619888, 0.317978813960745, 0.321944333142034, 0.325785467296709, 0.329497329826039, 0.333075806785722, 0.336517140069245, 0.339817307683451, 0.342971104478470, 0.345970673236043, 0.348802759621146, 0.351442311597064, 0.353830020283959, 0.354955038334893, 0.235996868936240, 0.237259747553113, 0.240358026746013, 0.243456167291127, 0.245029036669523, 0.245145336632522, 0.246553748583989, 0.249650365907212, 0.252745629835920, 0.255411570700446, 0.255527691192725, 0.255839165673704, 0.258930612916777, 0.262019624744252, 0.265105867532641, 0.268189020392870, 0.271268774728241, 0.274344833811948, 0.277416912382827, 0.280484736258203, 0.283548041962723, 0.286606576372236, 0.289660096371782, 0.292708368526901, 0.295751168767483, 0.298788282083481, 0.301819502231834, 0.304844631454020, 0.307863480203684, 0.310875866883849, 0.313881617593231, 0.316880565881230, 0.319872552511194, 0.322857425231579, 0.325835038554655, 0.328805253542435, 0.331767937599515, 0.334722964272544, 0.337670213056056, 0.340609569204406, 0.343540923549576, 0.346464172324634, 0.349379216992622, 0.352285964080678, 0.355184325019218, 0.358074215985969]
	# S_1_P2_DHV =  [0.291022057782900, 0.295951958184037, 0.300827160820633, 0.305632488542333, 0.310351893051314, 0.314969061154339, 0.319468210971551, 0.323834995319327, 0.328057371303902, 0.332126266755621, 0.336035898049091, 0.339783669013325, 0.343369677231056, 0.346795929576319, 0.350065393301933, 0.353180977390338, 0.356144461918853, 0.358955271644787, 0.361608772040304, 0.362996004843946, 0.363102108978563, 0.364093213531999, 0.366382444351142, 0.368409206845696, 0.369254443642157, 0.280974428245624, 0.282017155010234, 0.284337769022368, 0.286693532391806, 0.288749542801756, 0.288839638822677, 0.289081525161202, 0.291499089361689, 0.293943797306727, 0.296413424706242, 0.298905927735525, 0.301419423370539, 0.303952172438047, 0.306502564935414, 0.309069107258429, 0.311650411041466, 0.314245183366871, 0.316852218142476, 0.319470388480101, 0.322098639935410, 0.324735984491884, 0.327381495190093, 0.330034301318572, 0.332693584095140, 0.335358572777958, 0.338028541154270, 0.340702804362138, 0.343380716006572, 0.346061665536674, 0.348745075854830, 0.351430401132703, 0.354117124812011, 0.356804757770787, 0.359492836638208, 0.362180922243063, 0.364868598182742, 0.367555469501089, 0.370241161464831, 0.372925318427314, 0.375607602786093, 0.378287693983021]
	# S_1_P3_DHV =  [0.296037932340702, 0.301581175172076, 0.307137419685592, 0.312690031292753, 0.318214659292667, 0.323675969040932, 0.329024064935090, 0.334192707237204, 0.339104074058606, 0.343685371594877, 0.347893976806592, 0.351733218294751, 0.355244307057159, 0.358483728851476, 0.361504663717883, 0.364349052959828, 0.367046848807795, 0.369617933419464, 0.372074297283704, 0.373377433782964, 0.373477827175489, 0.374421628253627, 0.376660085736335, 0.378784093336241, 0.379814955600515, 0.379889817146639, 0.380780609381485, 0.382623948273740, 0.384258436882642, 0.385349208975200, 0.323315782316191, 0.323435194381678, 0.324696610481343, 0.326083367334617, 0.327580249512193, 0.329174544899525, 0.330855509800069, 0.332613971554761, 0.334442028011165, 0.336332816619509, 0.338280334506667, 0.340279296487738, 0.342325021727640, 0.344413342327578, 0.346540528893240, 0.348703229401508, 0.350898418586876, 0.353123355726980, 0.355375549191849, 0.357652726483330, 0.359952808763922, 0.362273889082024, 0.364614213660276, 0.366972165737500, 0.369346251551560, 0.371735088126690, 0.374137392589390, 0.376551972785267, 0.378977719008094, 0.381413596683766, 0.383858639877400, 0.386311945512709, 0.388772668204453, 0.391240015663211, 0.393713244488567, 0.396191656486459]

	T0 = [290.0, 290.6779661, 291.3559322, 292.03389831, 292.71186441, 293.38983051, 294.06779661, 294.74576271, 295.42372881, 296.10169492, 296.77966102, 297.45762712, 298.13559322, 298.81355932, 299.49152542, 299.69421925, 299.77421925, 300.16949153, 300.84745763, 301.52542373, 302.20338983, 302.88135593, 303.55932203, 304.23728814, 304.91525424, 305.59322034, 306.27118644, 306.94915254, 307.54260373, 307.61260373, 307.62711864, 308.30508475, 308.98305085, 309.66101695, 310.33898305, 311.01694915, 311.69491525, 312.37288136, 313.05084746, 313.72881356, 314.40677966, 315.08474576, 315.45711514, 315.52711514, 315.76271186, 316.44067797, 317.11864407, 317.79661017, 318.47457627, 319.15254237, 319.83050847, 320.50847458, 321.18644068, 321.86440678, 322.54237288, 323.22033898, 323.89830508, 324.57627119, 325.25423729, 325.93220339, 326.61016949, 327.28813559, 327.96610169, 328.6440678, 329.3220339, 330.0]
	S_1_P1_DHV =  [0.304456946618227, 0.305535439151210, 0.306610366445478, 0.307681572487271, 0.308748881756768, 0.309812094417380, 0.310870979703552, 0.311925266539211, 0.312974629697035, 0.314018668348726, 0.315056870618981, 0.316088549641070, 0.317112712398667, 0.318127727294431, 0.319129979723347, 0.319425265231469, 0.308433997032113, 0.308819780595928, 0.309484643880623, 0.310153451302620, 0.310826130050739, 0.311502608365221, 0.312182815528396, 0.312866681854758, 0.313554138680535, 0.314245118352825, 0.314939554218368, 0.315637380611998, 0.316250948281517, 0.316323487125867, 0.316338532844861, 0.317042947192401, 0.317750560882202, 0.318461312081684, 0.319175139885718, 0.319891984304171, 0.320611786249415, 0.321334487523820, 0.322060030807265, 0.322788359644665, 0.323519418433548, 0.324253152411690, 0.324657270658008, 0.324733326555533, 0.324989507644827, 0.325728431014442, 0.326469870205657, 0.327213773695222, 0.327960090739617, 0.328708771363274, 0.329459766346927, 0.330213027216077, 0.330968506229616, 0.331726156368562, 0.332485931324955, 0.333247785490884, 0.334011673947669, 0.334777552455181, 0.335545377441316, 0.336315105991617, 0.337086695839044, 0.337860105353893, 0.338635293533870, 0.339412219994307, 0.340190844958531, 0.340971129248381]
	S_1_P2_DHV =  [0.304604992900903, 0.305699988958015, 0.306792293327434, 0.307881839018114, 0.308968555442556, 0.310052367969980, 0.311133197403620, 0.312210959364869, 0.313285563561954, 0.314356912913956, 0.315424902491472, 0.316489418221828, 0.317550335287543, 0.318607516118541, 0.319660807836284, 0.319974933884319, 0.320098813458606, 0.320710038942611, 0.321755014941862, 0.322795512412578, 0.323831270747535, 0.324861980239833, 0.325887264144315, 0.326906650144002, 0.327919521534309, 0.328925024732612, 0.329921864507478, 0.330907704561730, 0.331756260944449, 0.328950917619284, 0.328965508260804, 0.329648130830616, 0.330332954889257, 0.331019990312008, 0.331709242197215, 0.332400711470861, 0.333094395402752, 0.333790288050117, 0.334488380641061, 0.335188661907786, 0.335891118377574, 0.336595734627996, 0.336983654225647, 0.337056649371084, 0.337302493511653, 0.338011376354816, 0.338722363133583, 0.339435432630585, 0.340150562574784, 0.340867729766499, 0.341586910189509, 0.342308079111767, 0.343031211176072, 0.343756280481851, 0.344483260659036, 0.345212124934913, 0.345942846194685, 0.346675397036410, 0.347409749820892, 0.348145876717038, 0.348883749743114, 0.349623340804304, 0.350364621726995, 0.351107564289850, 0.351852140252379, 0.352598321380823]
	S_1_P3_DHV =  [0.304654887299414, 0.305759919435512, 0.306862637959683, 0.307962998908640, 0.309060957028133, 0.310156465655678, 0.311249476590301, 0.312339939947410, 0.313427803996553, 0.314513014979379, 0.315595516904638, 0.316675251316362, 0.317752157030595, 0.318826169835022, 0.319897222144546, 0.320216852908515, 0.320342931028261, 0.320965242604252, 0.322030155629043, 0.323091880866508, 0.324150332565910, 0.325205418831352, 0.326257040730580, 0.327305091221908, 0.328349453849119, 0.329390001136410, 0.330426592589575, 0.331459072171344, 0.332359331425945, 0.332465298188345, 0.332487265060716, 0.333510973415233, 0.334529970708341, 0.335543993967305, 0.336552732803106, 0.337555813316722, 0.338552773359378, 0.339543022136488, 0.340525768660621, 0.341499879367877, 0.342463537189853, 0.343413065445634, 0.343924283651724, 0.343802976464552, 0.344068172783527, 0.344828409215007, 0.345585268912219, 0.346339610450463, 0.347092045343891, 0.347843033460844, 0.348592933651104, 0.349342033475209, 0.350090567977070, 0.350838732206380, 0.351586689938952, 0.352334579963732, 0.353082520746982, 0.353830613976509, 0.354578947310407, 0.355327596546645, 0.356076627361918, 0.356826096724098, 0.357576054053240, 0.358326542186038, 0.359077598184606, 0.359829254020497]

	if Isobars:

		if Plot_Phi:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(T0,phip_P1_DHV,'r',ls=ls1,label='phi_p_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phip_P2_DHV,'r',ls=ls2,label='phi_p_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phip_P3_DHV,'r',ls=ls3,label='phi_p_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,phis_P1_DHV,'m',ls=ls1,label='phi_s_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phis_P2_DHV,'m',ls=ls2,label='phi_s_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phis_P3_DHV,'m',ls=ls3,label='phi_s_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,Rtilde_P1_DHV,'b',ls=ls1,label='Rtilde_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Rtilde_P2_DHV,'b',ls=ls2,label='Rtilde_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Rtilde_P3_DHV,'b',ls=ls3,label='Rtilde_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,phip0_P1_DHV,'k',ls=ls1,label='phi_p0_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phip0_P2_DHV,'k',ls=ls2,label='phi_p0_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phip0_P3_DHV,'k',ls=ls3,label='phi_p0_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,phis0_P1_DHV,'y',ls=ls1,label='phi_s0_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phis0_P2_DHV,'y',ls=ls2,label='phi_s0_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phis0_P3_DHV,'y',ls=ls3,label='phi_s0_{} MPa'.format(P3),lw=linewidth)

			plt.xlabel('Temperature T (K)',fontsize=axis_size)
			plt.ylabel('phi',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(phi_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Entropy:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(T0,S_1_P1_DHV,'k',ls=ls2,label='Present Theory at {} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,S_1_P2_DHV,'k',ls=ls1,label='{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,S_1_P3_DHV,'k',ls=ls3,label='{} MPa'.format(P3),lw=linewidth)

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
			plt.ylabel('Entropy $S$ $($ $J/g.K)$',fontsize=axis_size)
			plt.legend(loc=2,fontsize=size,numpoints=1,frameon=False)
			# plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(TD_axes)
			plt.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.10,wspace=0.30,hspace=0.25)
			figS.savefig('./'+output_folder+r'\PS_CO2_Self_Grassia_02kilo_POST_THESIS_Paper4_11_12_Entropy_final_new_nsolve_dis'+img_extension,dpi=240)

#Show plot windows.
plt.show()

print 'T0 = ', T0
print 'S_1_P1_DHV = ', S_1_P1_DHV
print 'S_1_P2_DHV = ', S_1_P2_DHV
print 'S_1_P3_DHV = ', S_1_P3_DHV


# result = calculateBinarySolubilitySwelling('CHV',P,T,Mp,Ms,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,method='disparate',Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol,forward=forward,backward=backward)
# Xs = result[2]
# Sw = result[3]
# phip = result[4]
# phis = result[5]
# Rtilde = result[6]
# phip0 = result[7]
# phis0 = result[8]

# properties=calculateThermodynamicVariables(P,T,phip,phis,phip0,phis0,Mp,Ms,g=g,epsilon_p=epsilon_p,zeta=zeta,delta=delta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
# S_1 = properties[2]
# print 'S_1:', S_1,' x*S_infty:', x*S_infty













































































