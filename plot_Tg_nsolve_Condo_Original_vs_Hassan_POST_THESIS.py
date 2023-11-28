
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
from calculateBinaryVariablesCHV_Condo_nsolve import *

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
	# guess1 = npy.array([0.99,0.999,0.95,0.85,0.7,0.5,0.30,0.10,0.01,0.05,1.0])
	guess1 = npy.array([0.99,0.95,0.85])
	
	success = 0
	for i in range(len(guess1)):
		# print 'for loop of phip0:', i
		try:
			phip0 = nsolve(EOS_p0,phi_p,guess1[i],verify=True)
			# print phip0
		except:
			pass
		
		phip0 = complex(phip0)

		if phip0.real>0.0  and abs(phip0.real)<=1.0 and abs(phip0.imag)<=10E-3:
			# print 'Is phip0 complex:',phip0
			phip0 = abs(phip0)
			# phip0 = round(phip0, 6)
			# print Ppstar, Tpstar, Rpstar, Mp, P, T, phip0
			residual = EOS_pure_k(phip0,P,T,Mp,Ppstar,Tpstar,Rpstar)
			# print 'Hurry! phip0 is:', phip0, 'and residual is:', residual
			phip0_all_values.append(phip0)
			success = 1
			# break				#Do not break it because it is detecting phi-->0 as a possible solution. But, phi---> is correct solution
		else:
			# phip0 = 0.0
			pass
	
	if success == 0:
		print 'program failed to get phip0 hence we are appendig phip0 = NAN' 
		phip0_all_values.append(float('nan'))

	phip0_all_values = npy.array(remove_duplicates(phip0_all_values))
	# print phip0_all_values
	phip0 = phip0_all_values[0]

	for i in range(len(phip0_all_values)):
		if phip0_all_values[i]>phip0 and phip0<=1.0 and phip0>=0.0:
			phip0 = phip0_all_values[i]
		else:
			pass	

	print 'phip0_all_values are:', phip0_all_values, 'however chosen phip0 is:', phip0

	if phip0==0.0:
		print 'Program Failed to get value of phip0'
		# raise ValueError('Program Failed to get value of phip0')

	phis0 = 0.0
	phis0_all_values = []
	success = 0
	# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.000001,0.00001,0.90,0.95,0.97,0.98,0.99,0.999,0.9999,0.99999,0.999999])
	# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.65,0.75,0.85,0.00001,0.90,0.97,0.99,0.999,0.9999])
	# guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.30,0.45,0.70,0.90,0.95,0.999])
	guess2 = npy.array([0.90,0.80,0.50,0.01,0.10])
	Pstilde = P/Psstar
	Tstilde = T/Tsstar
	guess_low = ((Tstilde/alpha_s0)-sqrt((Tstilde/alpha_s0)**2-4*(1-(Tstilde/2))*Pstilde))/(2*(1-(Tstilde/2)))  #Correct Guess by quadratic approximation for smallest phis0
	# print guess_low
	# guess2 = npy.array([guess_low,guess_low+0.01,guess_low-0.01,0.010,0.020,0.030,0.040])
	# coeff = [(Tstilde/3), ((Tstilde/2)-1), (Tstilde/alpha_s0), -Pstilde]			#Correct Guess by cubic approximation for smallest phis0
	# coeff = [(Tstilde/4), (Tstilde/3), ((Tstilde/2)-1), (Tstilde/alpha_s0), -Pstilde]			#Correct Guess by forth power approximation for smallest phis0
	# coeff = [(Tstilde/5), (Tstilde/4), (Tstilde/3), ((Tstilde/2)-1), (Tstilde/alpha_s0), -Pstilde]			#Correct Guess by fifth power approximation for smallest phis0
	# coeff = [(Tstilde/6), (Tstilde/5), (Tstilde/4), (Tstilde/3), ((Tstilde/2)-1), (Tstilde/alpha_s0), -Pstilde]			#Correct Guess by sixth power approximation for smallest phis0
	coeff = [(Tstilde/7), (Tstilde/6), (Tstilde/5), (Tstilde/4), (Tstilde/3), ((Tstilde/2)-1), (Tstilde/alpha_s0), -Pstilde]			#Correct Guess by seventh power approximation for smallest phis0

	root_final = 999.0
	answer = npy.roots(coeff)
	# print answer
	for i in range(len(answer)):
		root = complex(answer[i])
		if root.real>0.0 and abs(root.imag)<=10E-3:
			if root_final>root.real:
				root_final = root.real
				picked = root_final
	if picked <=0.5:
		phis0_all_values.append(picked)
		success = 1
	# print 'phis0_all_values picked from cubic approximation is:', phis0_all_values


	for i in range(len(guess2)):
		# print 'for loop of phis0:', i

		try:
			phis0 = nsolve(EOS_s0,phi_s,guess2[i],verify=True)
		except:
			pass

		phis0 = complex(phis0)
		# print phis0
		if phis0.real>0.0 and abs(phis0.real)<=1.0 and abs(phis0.imag)<=10E-3:
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
			success = 1
			# break			#Do not break it because CO2 has multiple solution.
		else:
			# phis0 = 0.0
			pass

	# print phis0_all_values, 'jhjgjhgjkgj'
	if success == 0:
		print 'program failed to get phis0 hence we are appendig phis0 = NAN' 
		phis0_all_values.append(float('nan'))
	
	phis0_all_values = npy.array(remove_duplicates(phis0_all_values))


	# print phis0_all_values
	phis0 = phis0_all_values[0]

	for i in range(len(phis0_all_values)):
		if phis0_all_values[i]<phis0 and phis0_all_values[i]!=0.0 and phis0<=1.0 and phis0>=0.0 :			#Original is >
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
	# guess_phip = npy.array([0.40,0.60,0.80,0.90])
	# guess_phis = npy.array([0.01,0.10,0.30,0.50])

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
			
			if phip.real>0.0  and abs(phip.real<=1.0) and abs(phip.imag)<=10E-3 and phis.real>0.0  and abs(phis.real)<=1.0 and abs(phis.imag)<=10E-3:
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

	# print 'line number is:',get_linenumber()

	phip = phip_all_values[0]
	phis = phis_all_values[0]

	for i in range(len(phip_all_values)):
		if phip_all_values[i]>phip and phip<=1.0 and phip>=0.0 and phis<=1.0 and phis>=0.0:
			phip = phip_all_values[i]
			phis = phis_all_values[i]
		else:
			pass		

	print 'However chosen value of phip is:', phip
	print 'However chosen value of phis is:', phis

	#print 'line number is:',get_linenumber()
	if phip==0.0 or phis==0.0:
		print 'Program Failed to get value of phip and phis'
		phip = float('nan')
		phis = float('nan')
		# raise ValueError('Program Failed to get value of phip and phis')

	if phis0 == 'nan':
		print 'Program Failed to get value of phis0 so assigning NAN to phip and phis'
		phip = float('nan')
		phis = float('nan')
		# raise ValueError('Program Failed to get value of phip and phis')

	# phip,phis = nsolve([EOS_m,(mu_s_m-mu_s0.subs(phi_s,phis0))],[phi_p,phi_s],[0.85,0.05],verify=True)

	# print phip_all_values
	# print phis_all_values
	#print 'line number is:',get_linenumber()
	#print 'line number is:',get_linenumber()

	# print 'Returning phip is:', phip
	# print 'Returning phis is:', phis
	
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

	if hphip0 =='nan' or hphis0 =='nan' or hsol_phip =='nan' or hsol_phis =='nan':
		ms = float('nan')
		Sw = float('nan')
	
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
	max_Pg=11.0
	# step_Pg=1
	num_of_points = 12

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

###############################################################
################################################################
###  Condo_Original_Below:
################################################################

def calculate_Phi_Condo_Original(P0,T0,Mp,Ms,**kwargs):

	if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
		answer = binaryPhaseEquilibriumCondo_Original_nsolve(P0,T0,Mp,Ms,**kwargs)
		result = answer

	elif not isListOrNpyArray(T0) and isListOrNpyArray(P0):
		result = [[range(0,len(P0))] for x in range(6)]
		T = range(0,len(P0))
		Rtildem = range(0,len(P0))
		cphis = range(0,len(P0))
		Rtildep0 = range(0,len(P0))
		Rtildes0 = range(0,len(P0))

		for i in range(0,len(P0)):
			answer = binaryPhaseEquilibriumCondo_Original_nsolve(P0[i],T0,Mp,Ms,**kwargs)
			T[i] = answer[1]
			Rtildem[i] = answer[2]
			cphis[i] = answer[3]
			Rtildep0[i] = answer[4]
			Rtildes0[i] = answer[5]
			
		result[0] = P0
		result[1] = T
		result[2] = Rtildem
		result[3] = cphis
		result[4] = Rtildep0
		result[5] = Rtildes0

	elif not isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(6)]
		P = range(0,len(T0))
		Rtildem = range(0,len(T0))
		cphis = range(0,len(T0))
		Rtildep0 = range(0,len(T0))
		Rtildes0 = range(0,len(T0))

		for i in range(0,len(T0)):
			answer = binaryPhaseEquilibriumCondo_Original_nsolve(P0,T0[i],Mp,Ms,**kwargs)
			P[i] = answer[0]
			Rtildem[i] = answer[2]
			cphis[i] = answer[3]
			Rtildep0[i] = answer[4]
			Rtildes0[i] = answer[5]

		result[0] = P
		result[1] = T0
		result[2] = Rtildem
		result[3] = cphis
		result[4] = Rtildep0
		result[5] = Rtildes0
	
	elif isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(6)]
		P = range(0,len(T0))
		Rtildem = range(0,len(T0))
		cphis = range(0,len(T0))
		Rtildep0 = range(0,len(T0))
		Rtildes0 = range(0,len(T0))

		for i in range(0,len(T0)):
			answer = binaryPhaseEquilibriumCondo_Original_nsolve(P0[i],T0[i],Mp,Ms,**kwargs)
			P[i] = answer[0]
			Rtildem[i] = answer[2]
			cphis[i] = answer[3]
			Rtildep0[i] = answer[4]
			Rtildes0[i] = answer[5]

		result[0] = P
		result[1] = T0
		result[2] = Rtildem
		result[3] = cphis
		result[4] = Rtildep0
		result[5] = Rtildes0
	
	else:
		raise ValueError('In calculateBinarySwelling: Unknown error involving P0 and T0.')
	
	return result

def ThermodynamicVariables_Condo_Original(P,T,Rtilde,phi_s,Mp,Ms,**kwargs):
			
	phi_p=1.0-phi_s
	# print 'Temperature is', T

	for key,value in kwargs.items():
		exec "%s=%s" % (key,value)
	
	#PURE FLUID PARAMETERS.
	# vhp = kB*Tpstar/Ppstar
	# vhs = kB*Tsstar/Psstar
	rp = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	rs = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	
	# vhm=phi_s*vhs+phi_p*vhp
	
	# Tspstar=zeta*math.sqrt(Tsstar*Tpstar)
	# Xsp=(Tsstar+Tpstar-2*Tspstar)/T
	# Tstar=phi_s*Tsstar+phi_p*Tpstar-phi_s*phi_p*T*Xsp
	# Pstar=kB*Tstar/vhm

	# Ptilde=P/Pstar
	# Ttilde=T/Tstar

	r=1/(phi_s/rs+phi_p/rp)
	# print Rtilde
	
	vtilde=1/Rtilde

	# Pstilde=P/Psstar
	# Tstilde=T/Tsstar	
	# Pptilde=P/Ppstar
	# Tptilde=T/Tpstar

	Fp=((z-2)*exp(-epsilon_p/(kB*T)))/(1+(z-2)*exp(-epsilon_p/(kB*T)))
	epsilon_s = 0.0
	Fs=((z-2)*exp(-epsilon_s/(kB*T)))/(1+(z-2)*exp(-epsilon_s/(kB*T)))

	# print Fp, Fs, kB, Psstar, Tsstar, Rsstar, Ppstar, Tpstar, Rpstar, z, epsilon_p, rs, rp, r, phi_p, phi_s, Mp, Ms, Rtilde, P, T

	S_1=-1*((vtilde-1)*ln(1-Rtilde)+ln(Rtilde)/r+(phi_s/rs)*ln(phi_s/rs)+(phi_p/rp)*ln(phi_p/rp)+1+(ln(2/z)-1)/r+(phi_s/rs)*(rs-2)*(ln(1-Fs)-(Fs*epsilon_s/(kB*T)))+(phi_p/rp)*(rp-2)*(ln(1-Fp)-(Fp*epsilon_p/(kB*T))))
	S_2=(Ppstar/(Rpstar*Tpstar))*(1/phi_p)*(-1*((vtilde-1)*ln(1-Rtilde)+ln(Rtilde)/r+(phi_s/rs)*ln(phi_s/rs)+(phi_p/rp)*ln(phi_p/rp)+1+(ln(2/z)-1)/r+(phi_s/rs)*(rs-2)*(ln(1-Fs)-(Fs*epsilon_s/(kB*T)))+(phi_p/rp)*(rp-2)*(ln(1-Fp)-(Fp*epsilon_p/(kB*T)))))
	
	# print S_1
	# print S_2

	return [S_1,S_2]

def calculateThermodynamicVariables_Condo_Original(P0,T0,Rtilde,phi_s,Mp,Ms,**kwargs):

	if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
		# Rtilde=kphi_p+kphi_s
		# cphi_s=kphi_s/Rtilde
		TD = ThermodynamicVariables_Condo_Original(P0,T0,Rtilde,phi_s,Mp,Ms,**kwargs)
		result = TD
	
	elif not isListOrNpyArray(T0) and isListOrNpyArray(P0):
		result = [[range(0,len(P0))] for x in range(4)]
		T = range(0,len(P0))
		S_1 = range(0,len(P0))
		S_2 = range(0,len(P0))
	
		for i in range(0,len(P0)):

			# Rtilde[i]=kphi_p[i]+kphi_s[i]
			# cphi_s[i]=kphi_s[i]/Rtilde[i]
			TD = ThermodynamicVariables_Condo_Original(P0[i],T0,Rtilde[i],phi_s[i],Mp,Ms,**kwargs)
			T[i] = T0
			S_1[i] = TD[0]
			S_2[i] = TD[1]

		result[0] = P0
		result[1] = T
		result[2] = S_1
		result[3] = S_2

	elif not isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(4)]
		P = range(0,len(T0))
		S_1 = range(0,len(T0))
		S_2 = range(0,len(T0))
		# print Rtilde
		# print T0

		for i in range(0,len(T0)):
			# Rtilde[i]=kphi_p[i]+kphi_s[i]
			# cphi_s[i]=kphi_s[i]/Rtilde[i]
			TD = ThermodynamicVariables_Condo_Original(P0,T0[i],Rtilde[i],phi_s[i],Mp,Ms,**kwargs)
			P[i] = P0
			S_1[i] = TD[0]
			S_2[i] = TD[1]
	
		result[0] = P
		result[1] = T0
		result[2] = S_1
		result[3] = S_2
	
	elif isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(4)]
		P = range(0,len(T0))
		S_1 = range(0,len(T0))
		S_2 = range(0,len(T0))

		for i in range(0,len(T0)):
			# Rtilde[i]=kphi_p[i]+kphi_s[i]
			# cphi_s[i]=kphi_s[i]/Rtilde[i]
			TD = ThermodynamicVariables_Condo_Original(P0[i],T0[i],Rtilde[i],phi_s[i],Mp,Ms,**kwargs)
			S_1[i] = TD[0]
			S_2[i] = TD[1]

		result[0] = P0
		result[1] = T0
		result[2] = S_1
		result[3] = S_2
	
	else:
		raise ValueError('In ThermodynamicVariables_Condo_Original: Unknown error involving P0 and T0.')

	return result

def self_bisect_Tg_Condo_Original(T1,T2,P,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward):

	# prPurple('number_of_trails={}'.format(number_of_trails))
	criterion1 = Find_Tg_Bisect_Condo_Original(T1,P,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
	criterion2 = Find_Tg_Bisect_Condo_Original(T2,P,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)

	if (criterion1<0 and criterion2<0) or (criterion1>0 and criterion2>0):
		Tg = 0.0
		prGreen('Failed! Sign of both criterions are the same at T1={} and T2= {}'.format(T1,T2))

	if (criterion1>0 and criterion2<0) or (criterion1<0 and criterion2>0):
		prRed('Hurry! Different Sign found at T1={} and T2= {}'.format(T1,T2))
		Tg = bisect(Find_Tg_Bisect_Condo_Original,T1,T2,args=(P,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward),xtol=1E-2)
		# Tg = brentq(Find_Tg_Bisect_Condo_Original,T1,T2,args=(P,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward),xtol=1E-3)

	return Tg

def self_bisect_Pg_Condo_Original(P1,P2,T,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward):

	# prPurple('number_of_trails={}'.format(number_of_trails))
	criterion1 = Find_Pg_Bisect_Condo_Original(P1,T,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
	criterion2 = Find_Pg_Bisect_Condo_Original(P2,T,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)

	if (criterion1<0 and criterion2<0) or (criterion1>0 and criterion2>0):
		Pg = 0.0
		prGreen('Failed! Sign of both criterions are the same at P1={} and P2= {}'.format(P1,P2))

	if (criterion1>0 and criterion2<0) or (criterion1<0 and criterion2>0):
		prRed('Hurry! Different Sign found at P1={} and P2= {}'.format(P1,P2))
		Pg = bisect(Find_Pg_Bisect_Condo_Original,P1,P2,args=(T,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward),xtol=1E-2)
		# Pg = brentq(Find_Pg_Bisect_Condo_Original,P1,P2,args=(T,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward),xtol=1E-3)

	return Pg

def Find_Tg_Bisect_Condo_Original(Tg,P,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward):

	result = calculate_Phi_Condo_Original(P,Tg,Mp,Ms,zeta=zeta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol,forward=forward,backward=backward)
	Rtildem = result[2]
	cphis = result[3]
	Rtildep0 = result[4]
	Rtildes0 = result[5]

	properties=calculateThermodynamicVariables_Condo_Original(P,Tg,Rtildem,cphis,Mp,Ms,z=z,epsilon_p=epsilon_p,zeta=zeta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
	print properties
	S_1 = properties[0]
	S_2 = properties[1]

	criterion=S_1
	print 'S_1:', S_1
	# print 'P=',P,'T=',Tg,'phip=',phip,'phis=',phis,'phip0=',phip0,'phis0=',phis0,'Mp=',Mp,'Ms=',Ms,'g=',g,'epsilon_p=',epsilon_p,'zeta=',zeta,'delta=',delta,'Ppstar=',Ppstar,'Tpstar=',Tpstar,'Rpstar=',Rpstar,'Psstar=',Psstar,'Tsstar=',Tsstar,'Rsstar=',Rsstar

	return criterion

def Find_Pg_Bisect_Condo_Original(Pg,T,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward):

	result = calculate_Phi_Condo_Original(Pg,T,Mp,Ms,zeta=zeta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol,forward=forward,backward=backward)
	Rtildem = result[2]
	cphis = result[3]
	Rtildep0 = result[4]
	Rtildes0 = result[5]

	properties=calculateThermodynamicVariables_Condo_Original(Pg,T,Rtildem,cphis,Mp,Ms,z=z,epsilon_p=epsilon_p,zeta=zeta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
	S_1 = properties[0]
	S_2 = properties[1]
	print 'S_1:', S_1
	# print 'P=',P,'T=',Tg,'phip=',phip,'phis=',phis,'phip0=',phip0,'phis0=',phis0,'Mp=',Mp,'Ms=',Ms,'g=',g,'epsilon_p=',epsilon_p,'zeta=',zeta,'delta=',delta,'Ppstar=',Ppstar,'Tpstar=',Tpstar,'Rpstar=',Rpstar,'Psstar=',Psstar,'Tsstar=',Tsstar,'Rsstar=',Rsstar
	
	criterion=S_1

	return criterion

def GlassTemperature_Condo_Original(direction,P,Mp,Ms,**kwargs):
	
	for key,value in kwargs.items():
		exec "%s=%s" % (key,value)

	min_Tg=250.0
	max_Tg=360.0
	# step_Tg=10
	num_of_points = 15

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
		Tg= self_bisect_Tg_Condo_Original(T1,T2,P,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
		# try:
		# 	Tg= self_bisect_Tg_Condo_Original(T1,T2,P,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
		# except:
		# 	pass

		if Tg!=0.0:
			prRed('Hurry! Tg is:{} for direction {}'.format(Tg,direction))
			break
	if Tg==0.0:
		print 'Program Failed to get value of Tg in given bisect range in direction', direction

	return Tg

def GlassPressure_Condo_Original(direction,T,Mp,Ms,**kwargs):
	
	for key,value in kwargs.items():
		exec "%s=%s" % (key,value)

	min_Pg = 0.101325
	max_Pg = 11.0
	# step_Pg=1
	num_of_points = 12

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
		Pg = self_bisect_Pg_Condo_Original(P1,P2,T,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)

		# try:
		# 	Pg = self_bisect_Pg_Condo_Original(P1,P2,T,Mp,Ms,z,epsilon_p,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol,forward,backward)
		# except:
		# 	pass

		if Pg!=0.0:
			prRed('Hurry! Pg is:{} for direction {}'.format(Pg,direction))
			break
	if Pg==0.0:
		print 'Program Failed to get value of Pg in given bisect range in direction', direction

	return Pg



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
	Condo_Original=False 

	kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight,'Kier':Kier,'Hassan':Hassan,'Hassan_Var_Vol':Hassan_Var_Vol,'Condo':Condo,'Condo_Original':True}

	cepsilon_s,cepsilon_p,cz,czeta,epsilon_p,g,x,delta,zeta=Parameters_for_Mixtures_and_Tg(**kwargs)
	cdelta=100.0

	xS_infty=x*(Ppstar/(Tpstar*Rpstar))*(1+ln(1+g))
	S_infty=(Ppstar/(Tpstar*Rpstar))*(1+ln(1+g))
	# print Ppstar,Tpstar,Rpstar,g,epsilon_p,x,xS_infty

	forward=False		#Not 100% sure: Do not use forward=True and backward=False because: If forward=True and backward=False, then backward=False is penerating deep into the code and causing forward=True to not give any answers. i.e. all values are failing. 
	backward=True

	Find_Tg_at_P = False
	P=npy.linspace(0.101325,11.0,12)

	Find_Pg_at_T = True
	T_1 = npy.linspace(250,360,15)	
	T = npy.concatenate([T_1])

	Condo_Paper_Parameters = False
	if Condo_Paper_Parameters:

		#CO2 Parameters:
		Psstar = 574.0
		Tsstar = 308.64
		Rsstar = 1.505
		#PMMA Parameters:
		Ppstar = 503.0
		Tpstar = 696.0
		Rpstar = 1.269
		epsilon_p = 7443.0
		zeta = 1.1350
		z = 5.0		#Z should be written as float. Otherwise error will come
		# P0_X_T1 = [4.588776203,5.467284218,6.45164646,7.405418685,8.33077938,9.380496248,10.39087875]
		# T1 = 306.0
		# X0_X_T1 = [0.120498143,0.140490225,0.168289265,0.189996446,0.190033814,0.189966806,0.192712677]

		#PS Parameters:
		# Ppstar = 357.0
		# Tpstar = 735.0
		# Rpstar = 1.105
		# epsilon_p = 7151.0
		# zeta = 1.1240
		# z = 5.0		#Z should be written as float. Otherwise error will come

	zeta= 0.93
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


	if Condo_Original:

		kwargs = {'z':cz,'epsilon_p':cepsilon_p,'zeta':czeta,'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar,'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar,'Kier':Kier,'Hassan':Hassan,'Hassan_Var_Vol':Hassan_Var_Vol,'Condo':Condo,'forward':forward,'backward':backward}

		if Find_Tg_at_P:
			for i in range(0,len(P)):
				print 'Iterating for P:', P[i], 'for bisect method'
				if forward:
					Tg_bisect_fwd[i] = GlassTemperature_Condo_Original('fwd',P[i],Mp,Ms,**kwargs)
				if backward:
					Tg_bisect_bwd[i] = GlassTemperature_Condo_Original('bwd',P[i],Mp,Ms,**kwargs)

		if Find_Pg_at_T:
			for i in range(0,len(T)):
				print 'Iterating for T:', T[i], 'for bisect method'
				if forward:
					Pg_bisect_fwd[i] = GlassPressure_Condo_Original('fwd',T[i],Mp,Ms,**kwargs)
				if backward:
					Pg_bisect_bwd[i] = GlassPressure_Condo_Original('bwd',T[i],Mp,Ms,**kwargs)

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

	if Kier or Hassan or Condo or Hassan_Var_Vol or Condo_Original:
		if Find_Tg_at_P:
			if forward:
				print 'P before deleting zeros = ', P.tolist()
				print 'Tg_bisect_fwd before deleting zeros = ', Tg_bisect_fwd.tolist()
				P_fwd,Tg_bisect_fwd=discard_zeros(P,Tg_bisect_fwd)
				print 'P = ', P_fwd.tolist()
				print 'Tg_bisect_fwd = ', Tg_bisect_fwd.tolist()

			if backward:
				print 'P before deleting zeros = ', P.tolist()
				print 'Tg_bisect_bwd before deleting zeros = ', Tg_bisect_bwd.tolist()
				P_bwd,Tg_bisect_bwd=discard_zeros(P,Tg_bisect_bwd)
				print 'P = ', P_bwd.tolist()
				print 'Tg_bisect_bwd = ', Tg_bisect_bwd.tolist()

		if Find_Pg_at_T:
			if forward:
				print 'Pg_bisect_fwd before deleting zeros = ', Pg_bisect_fwd.tolist()
				print 'T  before deleting zeros = ', T.tolist()
				Pg_bisect_fwd,T_fwd=discard_zeros(Pg_bisect_fwd,T)
				print 'Pg_bisect_fwd = ', Pg_bisect_fwd.tolist()
				print 'T = ', T_fwd.tolist()
			if backward:
				print 'Pg_bisect_bwd before deleting zeros = ', Pg_bisect_bwd.tolist()
				print 'T before deleting zeros = ', T.tolist()
				Pg_bisect_bwd,T_bwd=discard_zeros(Pg_bisect_bwd,T)
				print 'Pg_bisect_bwd = ', Pg_bisect_bwd.tolist()
				print 'T = ', T_bwd.tolist()

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
	output_folder = 'NewResearch'

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

	if Kier or Hassan or Condo or Hassan_Var_Vol or Condo_Original:
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
	figPUREPS.savefig('./'+output_folder+r'\PS_CO2_Self_Grassia_02kilo_POST_THESIS_Paper4_11_12_Tg(P)'+img_extension,dpi=240)

	#Show plot windows.
	plt.show()



if True:	#For sorting and combining refined data
	#PMMA START:
	# Pg1 =  [4.1474473828125005, 4.7373148828125, 4.921648476562499, 4.9769485546875, 4.921648476562499, 4.7373148828125, 4.442381132812501, 4.0368472265625, 3.5391465234375, 2.9492790234375, 2.3041114453125, 1.6036437890625004, 0.9031761328124999, 0.2580085546874999]
	# T1 =  [276.1904761904762, 286.6666666666667, 291.9047619047619, 297.14285714285717, 302.3809523809524, 307.6190476190476, 312.8571428571429, 318.0952380952381, 323.3333333333333, 328.57142857142856, 333.8095238095238, 339.04761904761904, 344.2857142857143, 349.5238095238095]
	# Pg2 =  [4.394761621093751, 4.855595605468751, 4.96312353515625, 4.60981748046875, 3.85712197265625, 2.76648154296875, 1.4761463867187503, 0.20117236328125013]
	# T2 =  [280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0]
	# Pg3 =  [4.010733300781251, 4.14898349609375, 4.27187255859375, 4.394761621093751, 4.394761621093751, 4.67126201171875, 4.855595605468751, 4.96312353515625, 4.96312353515625, 4.96312353515625, 2.76648154296875]
	# T3 =  [274.0, 276.0, 278.0, 280.0, 280.0, 285.0, 290.0, 295.0, 300.0, 300.0, 330.0]
	# Pg4 =  [3.6171875, 3.6796875, 3.8984375, 3.9609375, 4.0234375, 4.0859375, 4.1484375, 4.2265625, 4.2734375, 4.3359375, 4.3984375, 4.3984375, 4.2890625]
	# T4 =  [268.57142857142856, 269.5238095238095, 272.3809523809524, 273.3333333333333, 274.2857142857143, 275.23809523809524, 276.1904761904762, 277.14285714285717, 278.0952380952381, 279.04761904761904, 280.0, 280.0, 315.0]
	# Pg5 =  [3.536458333333333, 3.6510416666666665, 3.6822916666666665, 3.713541666666667]
	# T5 =  [267.5, 269.1666666666667, 269.5833333333333, 270.0]
	# Pg6 =  [4.670857771381579, 4.8454895970394745, 4.942507277960527, 4.981314350328947, 4.923103741776316, 4.806682524671054, 4.612647162828948, 4.34099765625, 4.011137541118422, 3.6230668174342107, 3.1573819490131583, 2.6722935444078955, 2.1483980674342105, 1.585695518092106, 1.042396504934211, 0.49909749177631607]
	# T6 =  [285.0, 289.1666666666667, 293.3333333333333, 297.5, 301.6666666666667, 305.8333333333333, 310.0, 314.1666666666667, 318.3333333333333, 322.5, 326.6666666666667, 330.83333333333337, 335.0, 339.1666666666667, 343.33333333333337, 347.5]
	# Pg7 =  [0.101325, 0.12599375000000002, 0.1506625, 0.17533125, 0.2]
	# T7 =  [350.8671875, 350.6328125, 350.3984375, 350.1796875, 349.9609375]
	#PMMA END

	#PS START BELOW:
	# P1 =  [0.101325, 1.9177708333333334, 3.734216666666667, 5.5506625000000005, 7.367108333333334, 9.183554166666667, 11.0]
	# Tg1 =  [353.80712890625, 340.11083984375, 328.35888671875, 317.40185546875, 308.03466796875, 306.19775390625, 306.01513671875]
	# P2 =  [6.9362184836647724, 4.010295791903409, 1.4404377663352264]
	# Tg2 =  [310.0, 326.6666666666667, 343.33333333333337]
	# P3 =  [0.101325, 0.5554364583333333, 1.0095479166666668, 1.4636593750000002, 1.9177708333333334, 2.371882291666667, 2.8259937500000003, 3.280105208333334, 3.734216666666667, 4.188328125000001, 4.642439583333334, 5.096551041666667, 5.5506625000000005, 6.004773958333334, 6.458885416666668, 6.912996875000001, 7.367108333333334, 7.8212197916666675, 8.27533125, 8.729442708333334, 9.183554166666667, 9.637665625, 10.091777083333334, 10.545888541666667, 11.0]
	# Tg3 =  [353.806640625, 349.892578125, 346.447265625, 343.212890625, 340.107421875, 337.083984375, 334.130859375, 331.224609375, 328.365234375, 325.541015625, 322.775390625, 320.056640625, 317.396484375, 314.841796875, 312.392578125, 310.107421875, 308.033203125, 306.287109375, 306.322265625, 306.251953125, 306.193359375, 306.146484375, 306.099609375, 306.052734375, 306.017578125]
	# P4 =  [0.101325, 1.9177708333333334, 3.734216666666667, 5.5506625000000005, 7.367108333333334, 9.183554166666667, 11.0]
	# Tg4 =  [353.806640625, 340.107421875, 328.365234375, 317.396484375, 308.033203125, 306.193359375, 306.017578125]
	# P5 =  [6.0, 6.157894736842105, 6.315789473684211, 6.473684210526316, 6.631578947368421, 6.7894736842105265, 6.947368421052632, 7.105263157894736, 7.2631578947368425, 7.421052631578947, 7.578947368421053, 7.7368421052631575, 7.894736842105263, 8.052631578947368, 8.210526315789473, 8.368421052631579, 8.526315789473685, 8.68421052631579, 8.842105263157894, 9.0]
	# Tg5 =  [314.865234375, 313.998046875, 313.142578125, 312.310546875, 311.501953125, 310.705078125, 309.943359375, 309.193359375, 308.490234375, 307.810546875, 307.177734375, 306.580078125, 306.041015625, 305.572265625, 306.333984375, 306.310546875, 306.287109375, 306.263671875, 306.240234375, 306.216796875]
	# P6 =  [10.589751580255683, 7.741234250710227, 7.601904598721591, 7.462574946732955, 7.3387263671875, 7.214877787642046, 7.106510280539773, 6.982661700994318, 6.8742941938920445, 6.765926686789773, 6.673040252130682, 6.564672745028409, 6.456305237926136, 6.363418803267044, 6.270532368607954, 6.162164861505681, 6.06927842684659, 5.9763919921875]
	# Tg6 =  [306.05263157894734, 306.57894736842104, 307.10526315789474, 307.63157894736844, 308.1578947368421, 308.6842105263158, 309.2105263157895, 309.7368421052632, 310.2631578947368, 310.7894736842105, 311.3157894736842, 311.8421052631579, 312.36842105263156, 312.89473684210526, 313.42105263157896, 313.94736842105266, 314.4736842105263, 315.0]
	# P7 =  [8.0, 8.010416666666666, 8.020833333333334, 8.03125, 8.041666666666666, 8.052083333333334, 8.0625, 8.072916666666666, 8.083333333333334, 8.09375, 8.104166666666666, 8.114583333333334, 8.125, 8.135416666666666, 8.145833333333334, 8.15625, 8.166666666666666, 8.177083333333334, 8.1875, 8.197916666666666, 8.208333333333334, 8.21875, 8.229166666666666, 8.239583333333334, 8.25]
	# Tg7 =  [305.7277777777778, 305.69444444444446, 305.66111111111115, 305.6388888888889, 305.6055555555556, 305.58333333333337, 305.55000000000007, 305.5277777777778, 305.49444444444447, 305.4722222222223, 305.45000000000005, 305.46111111111117, 305.55000000000007, 305.65000000000003, 305.7611111111111, 305.86111111111114, 305.97222222222223, 306.0722222222222, 306.1833333333334, 306.2833333333334, 306.3388888888889, 306.32777777777784, 306.32777777777784, 306.32777777777784, 306.32777777777784]
	# PS END

	#PMMA ZETA = 0.93:
	P1 =  [6.0, 6.222222222222222, 6.444444444444445, 6.666666666666667, 6.888888888888889, 7.111111111111111, 7.333333333333333, 7.555555555555555, 7.777777777777778, 8.0]
	Tg1 =  [309.16259765625, 		306.11181640625, 301.91162109375, 288.41943359375, 291.33056640625, 294.12353515625, 296.75537109375, 296.62646484375, 296.50830078125, 296.39013671875]
	Pg2 =  [6.66607376953125, 6.71716130859375, 6.7512196679687495, 6.78527802734375, 6.81933638671875, 6.8704239257812505, 6.90448228515625, 6.938540644531251, 6.972599003906251, 7.0236865429687505, 7.05774490234375, 7.108832441406251, 7.142890800781251, 10.85525197265625, 9.76538447265625, 8.743633691406249, 7.789999628906252, 6.5638986914062505, 6.5638986914062505, 6.5638986914062505]
	T2 =  [288.5, 289.0, 289.5, 290.0, 290.5, 291.0, 291.5, 292.0, 292.5, 293.0, 293.5, 294.0, 294.5, 295.0, 295.5, 296.0, 296.5, 297.0, 297.5, 298.0]
	P3 =  [0.101325, 		1.3122888888888888, 2.523252777777778, 3.7342166666666667, 4.945180555555556, 6.156144444444445, 7.367108333333333, 8.578072222222222, 9.78903611111111, 11.0]
	Tg3 =  [351.02490234375, 342.96826171875, 	335.60986328125, 	327.97216796875, 	319.28173828125, 	307.10009765625, 296.73388671875, 296.08935546875, 295.48779296875, 294.93994140625]
	Pg4 =  [6.73419048828125, 6.5638986914062505, 6.29143181640625, 5.661352167968749, 4.75880564453125, 3.6519089648437504, 2.39174966796875, 1.0975320117187497]
	T4 =  [289.2857142857143, 297.1428571428571, 305.0, 			312.85714285714283, 320.7142857142857, 328.57142857142856, 336.42857142857144, 344.2857142857143]
	Pg5 =  [6.418750000000001, 6.48125, 6.54375, 6.60625, 6.66875, 6.73125, 6.79375, 6.86875, 6.93125, 6.99375, 6.56875, 6.56875, 6.58125, 6.56875, 6.56875]
	T5 =  [285.0, 285.85714285714283, 286.7142857142857, 287.57142857142856, 288.42857142857144, 289.2857142857143, 290.14285714285717, 291.0, 291.85714285714283, 292.7142857142857, 293.57142857142856, 294.42857142857144, 295.2857142857143, 296.14285714285717, 297.0]
	Pg6 =  [6.228947368421053, 6.386842105263158, 6.560526315789474, 6.449999999999999, 6.528947368421052, 6.560526315789474]
	T6 =  [282.3333333333333, 284.6666666666667, 287.0, 289.3333333333333, 291.6666666666667, 294.0]
	Pg7 =  [6.13421052631579, 6.213157894736842, 6.276315789473684, 6.339473684210526, 6.418421052631579, 6.481578947368422, 6.560526315789474, 6.386842105263158, 6.434210526315789, 6.465789473684211, 6.513157894736842, 6.528947368421052, 6.560526315789474, 6.560526315789474]
	T7 =  [281.0, 282.0, 283.0, 284.0, 285.0, 286.0, 287.0, 288.0, 289.0, 290.0, 291.0, 292.0, 293.0, 294.0]

	#PMMA ZETA = 0.92:
	# P1 =  [0.101325, 1.3122888888888888, 2.523252777777778, 3.7342166666666667, 4.945180555555556, 6.156144444444445, 7.367108333333333, 8.578072222222222, 9.78903611111111, 11.0]
	# Tg1 =  [351.13232421875, 344.01025390625, 337.70458984375, 331.44189453125, 324.94287109375, 317.87451171875, 309.81787109375, 306.26220703125, 305.76806640625, 305.33837890625]
	# P2 =  [6.938540644531251, 5.6783813476562495, 4.281988613281251, 2.7663916210937503, 1.2678238085937499]
	# Tg2 =  [312.85714285714283, 320.7142857142857, 328.57142857142856, 336.42857142857144, 344.2857142857143]
	# P3 =  [7.0, 7.166666666666667, 7.333333333333333, 7.5, 7.666666666666667, 7.833333333333333, 8.0, 8.166666666666666, 8.333333333333334, 8.5]
	# Tg3 =  [312.38525390625, 311.22509765625, 310.05419921875, 308.86181640625, 307.66943359375, 306.50927734375, 305.42431640625, 305.97216796875, 306.36962890625, 306.29443359375]
	# P4 =  [9.54400513671875, 7.824057988281252, 7.721882910156251, 7.619707832031251, 7.517532753906251, 7.415357675781251, 7.31318259765625, 7.22803669921875, 7.12586162109375, 7.006657363281251, 6.90448228515625]
	# Tg4 =  [305.85714285714283, 306.57142857142856, 307.2857142857143, 308.0, 308.7142857142857, 309.42857142857144, 310.14285714285717, 310.85714285714283, 311.57142857142856, 312.2857142857143, 313.0]

	#PMMA ZETA = 0.90:
	# P1 =  [0.101325, 1.0921136363636363, 2.082902272727273, 3.0736909090909093, 4.064479545454546, 5.055268181818183, 6.046056818181818, 7.036845454545455, 8.02763409090909, 9.018422727272727, 10.009211363636364, 11.0]
	# Tg1 =  [351.30419921875, 346.65283203125, 342.67822265625, 338.97216796875, 335.41650390625, 331.98974609375, 328.70263671875, 325.58740234375, 322.72998046875, 320.33447265625, 319.01318359375, 318.68017578125]
	# P2 =  [8.845808769531248, 6.08708166015625, 3.77111322265625, 1.67652412109375]
	# Tg2 =  [320.7142857142857, 328.57142857142856, 336.42857142857144, 344.2857142857143]
	# P3 =  [6.0, 6.454545454545454, 6.909090909090909, 7.363636363636363, 7.818181818181818, 8.272727272727273, 8.727272727272727, 9.181818181818182, 9.636363636363637, 10.09090909090909, 10.545454545454545, 11.0]
	# Tg3 =  [328.85302734375, 327.39208984375, 325.97412109375, 324.60986328125, 323.29931640625, 322.08544921875, 320.97900390625, 320.00146484375, 319.24951171875, 318.98095703125, 318.81982421875, 318.68017578125]
	# P4 =  [9.37371333984375, 8.845808769531248, 8.38602091796875, 7.977320605468751, 7.602678652343751, 7.22803669921875]
	# Tg4 =  [319.64285714285717, 320.7142857142857, 321.7857142857143, 322.85714285714283, 323.92857142857144, 325.0]

	#PMMA ZETA = 0.88:
	# P1 =  [0.101325, 1.0921136363636363, 2.082902272727273, 3.0736909090909093, 4.064479545454546, 5.055268181818183, 6.046056818181818, 7.036845454545455, 8.02763409090909, 9.018422727272727, 10.009211363636364, 11.0]
	# Tg1 =  [351.44384765625, 347.80224609375, 344.75146484375, 341.97998046875, 339.40185546875, 337.01708984375, 334.81494140625, 332.79541015625, 331.01220703125, 329.48681640625, 328.28369140625, 327.52099609375]
	# P2 =  [9.74835529296875, 5.3037393945312505, 2.23848705078125]
	# Tg2 =  [328.57142857142856, 336.42857142857144, 344.2857142857143]


	if Polymer_Type == 'PMMA':

		T = npy.concatenate([T1, T2, T3, T4, T5, T6, T7])
		Pg = npy.concatenate([Pg1, Pg2, Pg3, Pg4, Pg5, Pg6, Pg7])

		T = npy.array(T)
		Pg = npy.array(Pg)

		inds = T.argsort()
		sortedT = T[inds]
		sortedPg = Pg[inds]

		T = sortedT.tolist( )		#To convert array T to list T. In this way Print will give comma between elements
		Pg = sortedPg.tolist( )

		print 'T = ', T
		print 'Pg = ', Pg


	if Polymer_Type == 'PS':

		Tg = npy.concatenate([Tg1, Tg2])
		P = npy.concatenate([P1, P2])

		Tg = npy.array(Tg)
		P = npy.array(P)

		inds = P.argsort()
		sortedP = P[inds]
		sortedTg = Tg[inds]

		Tg = sortedTg.tolist( )		#To convert array T to list T. In this way Print will give comma between elements
		P = sortedP.tolist( )

		print 'Tg = ', Tg
		print 'P = ', P







# Tg_92 =  [351.13232421875, 344.2857142857143, 344.01025390625, 337.70458984375, 336.42857142857144, 331.44189453125, 328.57142857142856, 324.94287109375, 320.7142857142857, 317.87451171875, 313.0, 312.85714285714283, 312.38525390625, 312.2857142857143, 311.57142857142856, 311.22509765625, 310.85714285714283, 310.14285714285717, 310.05419921875, 309.81787109375, 309.42857142857144, 308.86181640625, 308.7142857142857, 308.0, 307.66943359375, 307.2857142857143, 306.57142857142856, 306.50927734375, 305.42431640625, 305.97216796875, 306.36962890625, 306.29443359375, 306.26220703125, 305.85714285714283, 305.76806640625, 305.33837890625]
# P_92 =  [0.101325, 1.2678238085937499, 1.3122888888888888, 2.523252777777778, 2.7663916210937503, 3.7342166666666667, 4.281988613281251, 4.945180555555556, 5.6783813476562495, 6.156144444444445, 6.90448228515625, 6.938540644531251, 7.0, 7.006657363281251, 7.12586162109375, 7.166666666666667, 7.22803669921875, 7.31318259765625, 7.333333333333333, 7.367108333333333, 7.415357675781251, 7.5, 7.517532753906251, 7.619707832031251, 7.666666666666667, 7.721882910156251, 7.824057988281252, 7.833333333333333, 8.0, 8.166666666666666, 8.333333333333334, 8.5, 8.578072222222222, 9.54400513671875, 9.78903611111111, 11.0]

# Tg_90 =  [351.30419921875, 346.65283203125, 344.2857142857143, 342.67822265625, 338.97216796875, 336.42857142857144, 335.41650390625, 331.98974609375, 328.85302734375, 328.70263671875, 328.57142857142856, 327.39208984375, 325.97412109375, 325.58740234375, 325.0, 324.60986328125, 323.92857142857144, 323.29931640625, 322.85714285714283, 322.72998046875, 322.08544921875, 321.7857142857143, 320.97900390625, 320.7142857142857, 320.7142857142857, 320.33447265625, 320.00146484375, 319.64285714285717, 319.24951171875, 319.01318359375, 318.98095703125, 318.81982421875, 318.68017578125, 318.68017578125]
# P_90 =  [0.101325, 1.0921136363636363, 1.67652412109375, 2.082902272727273, 3.0736909090909093, 3.77111322265625, 4.064479545454546, 5.055268181818183, 6.0, 6.046056818181818, 6.08708166015625, 6.454545454545454, 6.909090909090909, 7.036845454545455, 7.22803669921875, 7.363636363636363, 7.602678652343751, 7.818181818181818, 7.977320605468751, 8.02763409090909, 8.272727272727273, 8.38602091796875, 8.727272727272727, 8.845808769531248, 8.845808769531248, 9.018422727272727, 9.181818181818182, 9.37371333984375, 9.636363636363637, 10.009211363636364, 10.09090909090909, 10.545454545454545, 11.0, 11.0]

# Tg_88 =  [351.44384765625, 347.80224609375, 344.75146484375, 344.2857142857143, 341.97998046875, 339.40185546875, 337.01708984375, 336.42857142857144, 334.81494140625, 332.79541015625, 331.01220703125, 329.48681640625, 328.57142857142856, 328.28369140625, 327.52099609375]
# P_88 =  [0.101325, 1.0921136363636363, 2.082902272727273, 2.23848705078125, 3.0736909090909093, 4.064479545454546, 5.055268181818183, 5.3037393945312505, 6.046056818181818, 7.036845454545455, 8.02763409090909, 9.018422727272727, 9.74835529296875, 10.009211363636364, 11.0]

# T_93 =  [288.0, 289.0, 289.3333333, 290.0, 291.0, 291.6666667, 292.0, 293.0, 293.5714286, 294.0, 294.0, 294.4285714, 295.2857143, 296.1428571, 297.0, 297.0, 297.1428571, 297.5, 298.0, 301.9116211, 305.0, 306.1118164, 307.1000977, 309.1625977, 312.8571429, 319.2817383, 320.7142857, 327.972168, 328.5714286, 335.6098633, 336.4285714, 342.9682617, 344.2857143, 351.0249023]
# Pg_93 =  [6.386842105, 6.434210526, 6.45, 6.465789474, 6.513157895, 6.528947368, 6.528947368, 6.560526316, 6.56875, 6.560526316, 6.560526316, 6.56875, 6.58125, 6.56875, 6.563898691, 6.56875, 6.563898691, 6.563898691, 6.563898691, 6.444444444, 6.291431816, 6.222222222, 6.156144444, 6.0, 5.661352168, 4.945180556, 4.758805645, 3.734216667, 3.651908965, 2.523252778, 2.391749668, 1.312288889, 1.097532012, 0.101325]																																																

# Tg_93 =  [281.0, 282.0, 282.3333333, 283.0, 284.0, 284.6666667, 285.0, 285.0, 285.8571429, 286.0, 286.7142857, 287.0, 287.0, 287.5714286, 288.5, 288.4194336, 288.4285714, 289.0, 289.2857143, 289.2857143, 289.5, 290.0, 290.1428571, 290.5, 291.0, 291.0, 291.3305664, 291.5, 291.8571429, 292.0, 292.5, 292.7142857, 293.0, 293.5, 294.0, 294.1235352, 294.5, 296.7553711, 296.7338867, 296.6264648, 296.5083008, 296.5, 296.3901367, 296.0893555, 296.0, 295.5, 295.487793, 295.0, 294.9399414]
# P_93 =  [6.134210526, 6.213157895, 6.228947368, 6.276315789, 6.339473684, 6.386842105, 6.418421053, 6.41875, 6.48125, 6.481578947, 6.54375, 6.560526316, 6.560526316, 6.60625, 6.66607377, 6.666666667, 6.66875, 6.717161309, 6.73125, 6.734190488, 6.751219668, 6.785278027, 6.79375, 6.819336387, 6.86875, 6.870423926, 6.888888889, 6.904482285, 6.93125, 6.938540645, 6.972599004, 6.99375, 7.023686543, 7.057744902, 7.108832441, 7.111111111, 7.142890801, 7.333333333, 7.367108333, 7.555555556, 7.777777778, 7.789999629, 8.0, 8.578072222, 8.743633691, 9.765384473, 9.789036111, 10.85525197, 11.0]