
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
from plot_Tg_nsolve_Condo_Original_vs_Hassan_POST_THESIS import calculate_Phi_Condo_Original, calculateThermodynamicVariables_Condo_Original


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
	guess1 = npy.array([0.99,0.999,0.95,0.85,0.7,0.5,0.30,0.10,0.01,0.05,1.0])
	success = 0
	for i in range(len(guess1)):
		# print 'for loop of phip0:', i
		try:
			phip0 = nsolve(EOS_p0,phi_p,guess1[i],verify=True)
			print phip0
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
	print phip0_all_values
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
	guess2 = npy.array([0.0001,0.001,0.01,0.02,0.05,0.10,0.30,0.45,0.70,0.90,0.95,0.999])
	# guess2 = npy.array([0.65,0.85,0.95,0.45,0.30,0.01,0.10])
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

			print 'Is phip complex:',phip
			print 'Is phis complex:',phis
			
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
				break
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


################### Condo Original Below   ###################3

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

# zeta = 1.07136995 #+/- 0.04821750 (4.50%) (init = 0.95)
# delta = 1.23734449 #+/- 0.10480090 (8.47%) (init = 1.344)

# zeta = 0.92

print 'Psstar = ', Psstar,', Tsstar = ', Tsstar ,', Rsstar = ', Rsstar,', Ppstar = ', Ppstar,', Tpstar = ', Tpstar,', Rpstar = ', Rpstar,', zeta = ', zeta,', delta = ', delta, ', Mp = ', Mp, ', Ms = ', Ms, ', g = ', g,', epsilon_p = ', epsilon_p,', x = ', x,', xS_infty = ', xS_infty

Isotherms=False
Isobars=True
Entropy=True #Isobars
Plot_Phi=False #Isobars

forward=False		#Not 100% sure: Do not use forward=True and backward=False because: If forward=True and backward=False, then backward=False is penerating deep into the code and causing forward=True to not give any answers. i.e. all values are failing. 
backward=True

Plot_S2=False	


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
	cepsilon_p = 7443.0
	czeta = 1.115 #1.1350
	cz = 5.0		#Z should be written as float. Otherwise error will come
	# P0_X_T1 = [4.588776203,5.467284218,6.45164646,7.405418685,8.33077938,9.380496248,10.39087875]
	# T1 = 306.0
	# X0_X_T1 = [0.120498143,0.140490225,0.168289265,0.189996446,0.190033814,0.189966806,0.192712677]

	#PS Parameters:
	# Ppstar = 357.0
	# Tpstar = 735.0
	# Rpstar = 1.105
	# cepsilon_p = 7151.0
	# czeta = 1.1240
	# cz = 5.0		#Z should be written as float. Otherwise error will come

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
	# T0 = npy.linspace(min(T0_X),max(T0_X),10)
	T0_1 = npy.linspace(250,360,30)		#max: 1400000  #Small pressure ==> entropy max reaches at smaller temperature
	T0_2 = npy.linspace(290,310,2)	
	T0_3 = npy.linspace(310,320,2)
	T0_4 = npy.linspace(320,360,2)
	T0 = npy.concatenate([T0_1])
	P1=9.0	#P0_X_P1[0]#0.101325
	P2=5.0	#7.15	#30.0
	P3=6.0	#9.0	#50.0

if False:
	if Isotherms:
		if T1 != 0.0:
			P1_discontinuity = find_discontinuity_pressure(T1,Psstar,Tsstar,Rsstar,Ms)
			# print (math.isnan(P1_discontinuity)), P1_discontinuity
			if not math.isnan(P1_discontinuity):
				P0 = npy.append(P0,[P1_discontinuity-1.0,P1_discontinuity+0.001] )
				P0 = npy.sort(P0)
		if T2 != 0.0:
			P2_discontinuity = find_discontinuity_pressure(T2,Psstar,Tsstar,Rsstar,Ms)
			# print (math.isnan(P2_discontinuity)), P2_discontinuity
			if not math.isnan(P2_discontinuity):
				P0 = npy.append(P0,[P2_discontinuity-1.0,P2_discontinuity+0.001] )
				P0 = npy.sort(P0)
		if T3 != 0.0:
			P3_discontinuity = find_discontinuity_pressure(T3,Psstar,Tsstar,Rsstar,Ms)
			# print (math.isnan(P3_discontinuity)), P3_discontinuity
			if not math.isnan(P3_discontinuity):
				P0 = npy.append(P0,[P3_discontinuity-1.0,P3_discontinuity+0.001] )
				P0 = npy.sort(P0)


	if Isobars:
		if P1 != 0.0:
			T1_discontinuity = find_discontinuity_temperature(P1,Psstar,Tsstar,Rsstar,Ms)
			print (math.isnan(T1_discontinuity)), T1_discontinuity
			if not math.isnan(T1_discontinuity):
				T0 = npy.append(T0,[T1_discontinuity-0.5,T1_discontinuity+0.5] )
				T0 = npy.sort(T0)
		if P2 != 0.0:
			T2_discontinuity = find_discontinuity_temperature(P2,Psstar,Tsstar,Rsstar,Ms)
			# print (math.isnan(T2_discontinuity)), T2_discontinuity
			if not math.isnan(T2_discontinuity):
				T0 = npy.append(T0,[T2_discontinuity-1.0,T2_discontinuity+0.07] )
				T0 = npy.sort(T0)
		if P3 != 0.0:
			T3_discontinuity = find_discontinuity_temperature(P3,Psstar,Tsstar,Rsstar,Ms)
			# print (math.isnan(T3_discontinuity)), T3_discontinuity
			if not math.isnan(T3_discontinuity):
				T0 = npy.append(T0,[T3_discontinuity-1.0,T3_discontinuity+0.07] )
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


if Condo_Original:
	if Isotherms:
		isotherm_label=1
		for i in range(0,number_of_isotherm):
			source='''if T%s!=0.0:
				result = calculate_Phi_Condo_Original(P0,T%s,Mp,Ms,zeta=czeta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				Rtildem_T%s = npy.array(result[2])
				cphis_T%s = npy.array(result[3])
				Rtildep0_T%s = npy.array(result[4])
				Rtildes0_T%s = npy.array(result[5])

				if Entropy:
					properties=calculateThermodynamicVariables_Condo_Original(P0,T%s,Rtildem_T%s,cphis_T%s,Mp,Ms,z=cz,epsilon_p=cepsilon_p,zeta=czeta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
					S_1_T%s = npy.array(properties[0])
					S_2_T%s = npy.array(properties[1])'''
			
			exec source %(isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label,isotherm_label)

			isotherm_label=isotherm_label+1

			######################################

	if Isobars:
		isobar_label=1
		for i in range(0,number_of_isobar):
			source='''if P%s!=0.0:
				result = calculate_Phi_Condo_Original(P%s,T0,Mp,Ms,zeta=czeta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar,Kier=Kier,Hassan=Hassan,Condo=Condo,Hassan_Var_Vol=Hassan_Var_Vol, forward=forward,backward=backward)
				Rtildem_P%s = result[2]
				cphis_P%s = result[3]
				Rtildep0_P%s = result[4]
				Rtildes0_P%s = result[5]
				print Rtildem_P1
				print cphis_P1
				if Entropy:
					properties=calculateThermodynamicVariables_Condo_Original(P%s,T0,Rtildem_P%s,cphis_P%s,Mp,Ms,z=cz,epsilon_p=cepsilon_p,zeta=czeta,Ppstar=Ppstar,Tpstar=Tpstar,Rpstar=Rpstar,Psstar=Psstar,Tsstar=Tsstar,Rsstar=Rsstar)
					S_1_P%s = properties[2]
					S_2_P%s = properties[3]'''
			
			exec source %(isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label,isobar_label)
			isobar_label=isobar_label+1
		
			########################################


#Setting saved image properties
img_extension = '.png'
img_dpi = None
output_folder = 'NewResearch'

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

	# T0 = [290.0, 290.6779661, 291.3559322, 292.03389831, 292.71186441, 293.38983051, 294.06779661, 294.74576271, 295.42372881, 296.10169492, 296.77966102, 297.45762712, 298.13559322, 298.81355932, 299.49152542, 299.69421925, 299.77421925, 300.16949153, 300.84745763, 301.52542373, 302.20338983, 302.88135593, 303.55932203, 304.23728814, 304.91525424, 305.59322034, 306.27118644, 306.94915254, 307.54260373, 307.61260373, 307.62711864, 308.30508475, 308.98305085, 309.66101695, 310.33898305, 311.01694915, 311.69491525, 312.37288136, 313.05084746, 313.72881356, 314.40677966, 315.08474576, 315.45711514, 315.52711514, 315.76271186, 316.44067797, 317.11864407, 317.79661017, 318.47457627, 319.15254237, 319.83050847, 320.50847458, 321.18644068, 321.86440678, 322.54237288, 323.22033898, 323.89830508, 324.57627119, 325.25423729, 325.93220339, 326.61016949, 327.28813559, 327.96610169, 328.6440678, 329.3220339, 330.0]
	# S_1_P1_DHV =  [0.304456946618227, 0.305535439151210, 0.306610366445478, 0.307681572487271, 0.308748881756768, 0.309812094417380, 0.310870979703552, 0.311925266539211, 0.312974629697035, 0.314018668348726, 0.315056870618981, 0.316088549641070, 0.317112712398667, 0.318127727294431, 0.319129979723347, 0.319425265231469, 0.308433997032113, 0.308819780595928, 0.309484643880623, 0.310153451302620, 0.310826130050739, 0.311502608365221, 0.312182815528396, 0.312866681854758, 0.313554138680535, 0.314245118352825, 0.314939554218368, 0.315637380611998, 0.316250948281517, 0.316323487125867, 0.316338532844861, 0.317042947192401, 0.317750560882202, 0.318461312081684, 0.319175139885718, 0.319891984304171, 0.320611786249415, 0.321334487523820, 0.322060030807265, 0.322788359644665, 0.323519418433548, 0.324253152411690, 0.324657270658008, 0.324733326555533, 0.324989507644827, 0.325728431014442, 0.326469870205657, 0.327213773695222, 0.327960090739617, 0.328708771363274, 0.329459766346927, 0.330213027216077, 0.330968506229616, 0.331726156368562, 0.332485931324955, 0.333247785490884, 0.334011673947669, 0.334777552455181, 0.335545377441316, 0.336315105991617, 0.337086695839044, 0.337860105353893, 0.338635293533870, 0.339412219994307, 0.340190844958531, 0.340971129248381]
	# S_1_P2_DHV =  [0.304604992900903, 0.305699988958015, 0.306792293327434, 0.307881839018114, 0.308968555442556, 0.310052367969980, 0.311133197403620, 0.312210959364869, 0.313285563561954, 0.314356912913956, 0.315424902491472, 0.316489418221828, 0.317550335287543, 0.318607516118541, 0.319660807836284, 0.319974933884319, 0.320098813458606, 0.320710038942611, 0.321755014941862, 0.322795512412578, 0.323831270747535, 0.324861980239833, 0.325887264144315, 0.326906650144002, 0.327919521534309, 0.328925024732612, 0.329921864507478, 0.330907704561730, 0.331756260944449, 0.328950917619284, 0.328965508260804, 0.329648130830616, 0.330332954889257, 0.331019990312008, 0.331709242197215, 0.332400711470861, 0.333094395402752, 0.333790288050117, 0.334488380641061, 0.335188661907786, 0.335891118377574, 0.336595734627996, 0.336983654225647, 0.337056649371084, 0.337302493511653, 0.338011376354816, 0.338722363133583, 0.339435432630585, 0.340150562574784, 0.340867729766499, 0.341586910189509, 0.342308079111767, 0.343031211176072, 0.343756280481851, 0.344483260659036, 0.345212124934913, 0.345942846194685, 0.346675397036410, 0.347409749820892, 0.348145876717038, 0.348883749743114, 0.349623340804304, 0.350364621726995, 0.351107564289850, 0.351852140252379, 0.352598321380823]
	# S_1_P3_DHV =  [0.304654887299414, 0.305759919435512, 0.306862637959683, 0.307962998908640, 0.309060957028133, 0.310156465655678, 0.311249476590301, 0.312339939947410, 0.313427803996553, 0.314513014979379, 0.315595516904638, 0.316675251316362, 0.317752157030595, 0.318826169835022, 0.319897222144546, 0.320216852908515, 0.320342931028261, 0.320965242604252, 0.322030155629043, 0.323091880866508, 0.324150332565910, 0.325205418831352, 0.326257040730580, 0.327305091221908, 0.328349453849119, 0.329390001136410, 0.330426592589575, 0.331459072171344, 0.332359331425945, 0.332465298188345, 0.332487265060716, 0.333510973415233, 0.334529970708341, 0.335543993967305, 0.336552732803106, 0.337555813316722, 0.338552773359378, 0.339543022136488, 0.340525768660621, 0.341499879367877, 0.342463537189853, 0.343413065445634, 0.343924283651724, 0.343802976464552, 0.344068172783527, 0.344828409215007, 0.345585268912219, 0.346339610450463, 0.347092045343891, 0.347843033460844, 0.348592933651104, 0.349342033475209, 0.350090567977070, 0.350838732206380, 0.351586689938952, 0.352334579963732, 0.353082520746982, 0.353830613976509, 0.354578947310407, 0.355327596546645, 0.356076627361918, 0.356826096724098, 0.357576054053240, 0.358326542186038, 0.359077598184606, 0.359829254020497]

	# T0 =  [250.0, 250.0, 250.51724137931035, 251.0344827586207, 251.42857142857142, 251.55172413793105, 252.06896551724137, 252.58620689655172, 252.85714285714286, 253.10344827586206, 253.6206896551724, 254.13793103448276, 254.28571428571428, 254.6551724137931, 255.17241379310346, 255.68965517241378, 255.71428571428572, 256.2068965517241, 256.7241379310345, 257.14285714285717, 257.2413793103448, 257.7586206896552, 258.2758620689655, 258.57142857142856, 258.7931034482759, 259.3103448275862, 259.82758620689657, 260.0, 260.3448275862069, 260.86206896551727, 261.37931034482756, 261.42857142857144, 261.8965517241379, 262.41379310344826, 262.85714285714283, 262.9310344827586, 263.44827586206895, 263.9655172413793, 264.2857142857143, 264.48275862068965, 265.0, 265.0, 265.1515151515151, 265.3030303030303, 265.45454545454544, 265.6060606060606, 265.7142857142857, 265.75757575757575, 265.90909090909093, 266.06060606060606, 266.2121212121212, 266.3636363636364, 266.5151515151515, 266.6666666666667, 266.8181818181818, 266.969696969697, 267.1212121212121, 267.14285714285717, 267.27272727272725, 267.42424242424244, 267.57575757575756, 267.72727272727275, 267.8787878787879, 268.030303030303, 268.1818181818182, 268.3333333333333, 268.4848484848485, 268.57142857142856, 268.6363636363636, 268.7878787878788, 268.93939393939394, 269.09090909090907, 269.24242424242425, 269.3939393939394, 269.54545454545456, 269.6969696969697, 269.8484848484849, 270.0, 270.0, 270.1515151515151, 270.3030303030303, 270.45454545454544, 270.6060606060606, 270.75757575757575, 270.90909090909093, 271.06060606060606, 271.2121212121212, 271.3636363636364, 271.42857142857144, 271.5151515151515, 271.6666666666667, 271.8181818181818, 271.969696969697, 272.1212121212121, 272.27272727272725, 272.42424242424244, 272.57575757575756, 272.72727272727275, 272.85714285714283, 272.8787878787879, 273.030303030303, 273.1818181818182, 273.3333333333333, 273.4848484848485, 273.6363636363636, 273.7878787878788, 273.93939393939394, 274.09090909090907, 274.24242424242425, 274.2857142857143, 274.3939393939394, 274.54545454545456, 274.6969696969697, 274.8484848484849, 275.0, 275.1515151515151, 275.3030303030303, 275.45454545454544, 275.6060606060606, 275.7142857142857, 275.75757575757575, 275.90909090909093, 276.06060606060606, 276.2121212121212, 276.3636363636364, 276.5151515151515, 276.6666666666667, 276.8181818181818, 276.969696969697, 277.1212121212121, 277.14285714285717, 277.27272727272725, 277.42424242424244, 277.57575757575756, 277.72727272727275, 277.8787878787879, 278.030303030303, 278.1818181818182, 278.3333333333333, 278.4848484848485, 278.57142857142856, 278.6363636363636, 278.7878787878788, 278.93939393939394, 279.09090909090907, 279.24242424242425, 279.3939393939394, 279.54545454545456, 279.6969696969697, 279.8484848484849, 280.0, 280.0, 280.0, 280.40816326530614, 280.81632653061223, 281.2244897959184, 281.42857142857144, 281.6326530612245, 282.0408163265306, 282.44897959183675, 282.85714285714283, 282.8571428571429, 283.265306122449, 283.6734693877551, 284.0816326530612, 284.2857142857143, 284.48979591836735, 284.8979591836735, 285.3061224489796, 285.7142857142857, 285.7142857142857, 286.1224489795918, 286.53061224489795, 286.9387755102041, 287.14285714285717, 287.3469387755102, 287.7551020408163, 288.16326530612247, 288.57142857142856, 288.57142857142856, 288.9795918367347, 289.38775510204084, 289.7959183673469, 290.0, 290.2040816326531, 290.61224489795916, 291.0204081632653, 291.42857142857144, 291.42857142857144, 291.83673469387753, 292.2448979591837, 292.6530612244898, 292.8571428571429, 293.0612244897959, 293.46938775510205, 293.8775510204082, 294.2857142857143, 294.2857142857143, 294.6938775510204, 295.1020408163265, 295.51020408163265, 295.7142857142857, 295.9183673469388, 296.3265306122449, 296.734693877551, 297.14285714285717, 297.14285714285717, 297.55102040816325, 297.9591836734694, 298.36734693877554, 298.57142857142856, 298.7755102040816, 299.18367346938777, 299.59183673469386, 300.0, 300.0, 300.0, 301.42857142857144, 302.0689655172414, 302.8571428571429, 304.13793103448273, 304.2857142857143, 305.7142857142857, 306.2068965517241, 307.14285714285717, 308.2758620689655, 308.57142857142856, 310.0, 310.3448275862069, 311.42857142857144, 312.41379310344826, 312.8571428571429, 314.2857142857143, 314.48275862068965, 315.7142857142857, 316.55172413793105, 317.1428571428571, 318.57142857142856, 318.62068965517244, 320.0, 320.0, 320.6896551724138, 320.81632653061223, 321.6326530612245, 322.44897959183675, 322.7586206896552, 323.265306122449, 324.0816326530612, 324.82758620689657, 324.8979591836735, 325.7142857142857, 326.53061224489795, 326.8965517241379, 327.3469387755102, 328.16326530612247, 328.9655172413793, 328.9795918367347, 329.7959183673469, 330.61224489795916, 331.0344827586207, 331.42857142857144, 332.2448979591837, 333.0612244897959, 333.1034482758621, 333.8775510204082, 334.6938775510204, 335.17241379310343, 335.51020408163265, 336.3265306122449, 337.14285714285717, 337.2413793103448, 337.9591836734694, 338.7755102040816, 339.3103448275862, 339.59183673469386, 340.40816326530614, 341.2244897959184, 341.37931034482756, 342.0408163265306, 342.85714285714283, 343.44827586206895, 343.6734693877551, 344.48979591836735, 345.3061224489796, 345.51724137931035, 346.1224489795918, 346.9387755102041, 347.58620689655174, 347.7551020408163, 348.57142857142856, 349.38775510204084, 349.65517241379314, 350.2040816326531, 351.0204081632653, 351.7241379310345, 351.83673469387753, 352.6530612244898, 353.46938775510205, 353.7931034482759, 354.2857142857143, 355.1020408163265, 355.86206896551727, 355.9183673469388, 356.734693877551, 357.55102040816325, 357.9310344827586, 358.36734693877554, 359.18367346938777, 360.0, 360.0]
	# S_1_P1_DHV =  [0.179067457338083, 0.179067457338083, 0.179896828207601, 0.180727175685738, 0.181360464329123, 0.181558479468713, 0.18239071950633, 0.183223875997577, 0.18366065047718, 0.184057929386331, 0.184892860357178, 0.185728649831339, 0.185967601865132, 0.186565278962702, 0.187402729133951, 0.188240981952788, 0.188280918464127, 0.189080019248264, 0.189919823067179, 0.190600213501384, 0.190760375670587, 0.191601659530381, 0.192443657325959, 0.192925112870649, 0.193286351940967, 0.194129726460129, 0.19497376416614, 0.195255254581725, 0.195818448536642, 0.196663763241262, 0.197509692138641, 0.197590288245399, 0.198356219274027, 0.199203328875682, 0.199929874592783, 0.200051005353016, 0.200899233293536, 0.201747997460348, 0.202273685020538, 0.20259728278964, 0.20344707438821, 0.20344707438821, 0.203696096976233, 0.203945161373979, 0.204194267215251, 0.204443414135065, 0.204621401169214, 0.204692601769638, 0.204941829756394, 0.20519109773395, 0.205440405342113, 0.205689752221881, 0.205939138015429, 0.206188562366112, 0.206438024918455, 0.206687525318154, 0.206937063212063, 0.206972714522835, 0.207186638248198, 0.207436250075727, 0.207685898344966, 0.207935582707377, 0.208185302815562, 0.208435058323255, 0.208684848885325, 0.208934674157765, 0.20918453379769, 0.209327326033518, 0.209434427463333, 0.20968435481404, 0.209934315510264, 0.210184309213566, 0.210434335586605, 0.210684394293133, 0.210934484997999, 0.211184607367136, 0.211434761067561, 0.211684945767369, 0.211684945767369, 0.21193516113573, 0.212185406842887, 0.212435682560147, 0.212685987959879, 0.212936322715513, 0.213186686501531, 0.213437078993466, 0.213687499867899, 0.213937948802451, 0.214045292570279, 0.214188425475782, 0.214438929567588, 0.214689460758595, 0.214940018730554, 0.215190603166242, 0.215441213749452, 0.215691850164995, 0.215942512098691, 0.21619319923737, 0.216408093752119, 0.216443911268864, 0.216694647882007, 0.216945408766628, 0.217196193613549, 0.217447002114584, 0.217697833962528, 0.217948688851162, 0.218199566475242, 0.218450466530502, 0.218701388713645, 0.218773084787944, 0.218952332722342, 0.219203298255228, 0.2194542850119, 0.219705292692911, 0.219956320999767, 0.220207369634926, 0.220458438301792, 0.220709526704712, 0.220960634548973, 0.221140009034966, 0.2212117615408, 0.221462907387349, 0.221714071796708, 0.221965254477891, 0.222216455140836, 0.222467673496399, 0.222718909256354, 0.222970162133387, 0.223221431841103, 0.223472718094, 0.223508617464149, 0.223724020607489, 0.22397533909788, 0.22422667328238, 0.224478022879091, 0.224729387607006, 0.224980767186007, 0.225232161336859, 0.22548356978121, 0.225734992241589, 0.225878668405376, 0.225986428441397, 0.226237878104912, 0.226489340957278, 0.226740816724509, 0.226992305133481, 0.227243805911931, 0.227495318788455, 0.227746843492503, 0.227998379754378, 0.22824992730523, 0.22824992730523, 0.22824992730523, 0.228927619689436, 0.229605386846871, 0.230283223612624, 0.23062216649652, 0.230961124865395, 0.231639085527101, 0.232317100562486, 0.232995164978739, 0.232995164978739, 0.23367327382511, 0.234351422192544, 0.235029605213301, 0.235368708208719, 0.235707818060605, 0.236386055948269, 0.237064314130355, 0.237742587878083, 0.237742587878083, 0.238420872593127, 0.239099163580004, 0.239777456273001, 0.240116601835952, 0.240455746122208, 0.241134028615919, 0.2418122992803, 0.242490553679075, 0.242490553679075, 0.243168787413201, 0.243846996120559, 0.244525175475641, 0.244864252804293, 0.245203321189244, 0.24588142900817, 0.24655949471492, 0.247237514127382, 0.247237514127382, 0.247915483098648, 0.2485933975165, 0.249271253303347, 0.249610157945265, 0.249949046415835, 0.250626772844586, 0.251304428613921, 0.25198200978159, 0.25198200978159, 0.252659512438499, 0.253336932708445, 0.254014266747849, 0.254352900238836, 0.254691510745495, 0.255368660922274, 0.256045713530925, 0.256722664855783, 0.256722664855783, 0.257399511212528, 0.25807624894794, 0.258752874439649, 0.259091143970501, 0.259429384095895, 0.260105774355291, 0.260782041686577, 0.261458182588395, 0.261458182588395, 0.261458182588395, 0.263823630255046, 0.26488344805696, 0.266187340933587, 0.268304936631702, 0.268549172886374, 0.270908988560722, 0.271722226286569, 0.273266654478498, 0.275134912974034, 0.275622041129382, 0.277975022867604, 0.278542609947439, 0.280325477811991, 0.281944947114926, 0.282673287749171, 0.285018338039781, 0.285341570422798, 0.287360517527323, 0.28873214126631, 0.289699718451054, 0.292035836358263, 0.292116335926102, 0.294368770023348, 0.294368770023348, 0.29549384502866, 0.295700407442556, 0.297030955178228, 0.298360395704099, 0.298864373029291, 0.299688711811648, 0.301015886605574, 0.302227637716288, 0.302341903499355, 0.303666746210858, 0.304990398758037, 0.305583369735024, 0.306312845454687, 0.307634070906262, 0.30893131213086, 0.308954060005762, 0.310272797929685, 0.311590270134029, 0.312271219909812, 0.31290646235037, 0.31422136058198, 0.315534951100021, 0.315602859616036, 0.316847220439785, 0.31815815539699, 0.318926008925237, 0.319467743024131, 0.320775970626888, 0.322082825760582, 0.322240456253206, 0.32338829622668, 0.324692370069359, 0.325546000378721, 0.325995035572107, 0.327296281254385, 0.328596095868327, 0.328842450080125, 0.32989446839549, 0.331191388043651, 0.332129623784937, 0.332486844243646, 0.333780826646257, 0.335073325119136, 0.335407349231888, 0.33636432974378, 0.337653830812542, 0.338675463144839, 0.338941818825683, 0.340228284488468, 0.341513218708299, 0.341933810918041, 0.34279661259189, 0.344078457442474, 0.345182246312274, 0.345358744757056, 0.346637466223701, 0.34791461371885, 0.34842063116138, 0.349190179304689, 0.350464155226535, 0.351648835088786, 0.351736533910271, 0.353007307959808, 0.354276470154582, 0.354866735233613, 0.355544013447088, 0.356809930960439, 0.358074215985969, 0.358074215985969]
	# S_1_P2_DHV =  [0.257658334077234, 0.257658334077234, 0.256922942143912, 0.256284411886999, 0.255855808791684, 0.255731351327108, 0.2552543664776, 0.254845607646202, 0.254656455463316, 0.254498439558214, 0.254207196447068, 0.253966996871526, 0.25390713639281, 0.253773601415859, 0.253623301761994, 0.253512833099175, 0.253508515614807, 0.253439304156461, 0.253400140723897, 0.253392019517903, 0.253393039625437, 0.253415930881412, 0.253466946353974, 0.253508067352293, 0.25354439357292, 0.253646733737092, 0.253772563108431, 0.25381950087915, 0.253920597183164, 0.25408965713698, 0.254278658259145, 0.254297661280222, 0.254486599746255, 0.254712556126226, 0.25491991947559, 0.254955669579443, 0.255215143296854, 0.255490235655769, 0.255668053436319, 0.255780255057293, 0.256084555356615, 0.256084555356615, 0.256176309663539, 0.256269222680133, 0.25636328011312, 0.256458467932239, 0.256527143666158, 0.256554772367441, 0.256652179883424, 0.256750677230625, 0.256850251357118, 0.256950889448769, 0.25705257893313, 0.257155307452701, 0.257259062865988, 0.257363833242306, 0.25746960685673, 0.257484798595636, 0.257576372185212, 0.257684117899829, 0.257792832864185, 0.257902506128959, 0.258013126927538, 0.258124684671847, 0.258237168948246, 0.258350569513569, 0.258464876291273, 0.258530597499046, 0.258580079367694, 0.258696168988416, 0.258813135554738, 0.258930969620239, 0.259049661887444, 0.259169203204575, 0.259289584562402, 0.25941079709117, 0.259532832057616, 0.259655680862066, 0.259655680862066, 0.259779335035609, 0.259903786237344, 0.26002902625171, 0.260155046985869, 0.260281840467112, 0.260409398840773, 0.260537714366996, 0.260666779419166, 0.260796586481339, 0.260852443386098, 0.260927128146555, 0.26105839711298, 0.261190386183709, 0.261323088263201, 0.261456496358725, 0.261590603572384, 0.261725403103795, 0.261860888247569, 0.261997052389764, 0.262114299992286, 0.262133889007805, 0.262271391668704, 0.262409554025797, 0.262548369818679, 0.262687832871617, 0.262827937090051, 0.262968676462371, 0.263110045053998, 0.263252037011683, 0.263394646552907, 0.263435504772646, 0.263537867982555, 0.263681695663736, 0.263826124042216, 0.263971147632494, 0.264116761019055, 0.26426295885513, 0.264409735861482, 0.264557086825222, 0.264705006598648, 0.264811009011975, 0.264853490098107, 0.265002532302896, 0.265152128254167, 0.265302273053873, 0.26545296186373, 0.2656041899042, 0.265755952453496, 0.265908244846617, 0.266061062474392, 0.26621440078255, 0.26623634849659, 0.26636825527081, 0.266522621491991, 0.266677495051136, 0.266832871604657, 0.266988746859504, 0.267145116572339, 0.267301976548734, 0.267459322642399, 0.267617150754382, 0.267707553101932, 0.267775456832347, 0.267934236869815, 0.26809348690545, 0.268253203022342, 0.26841338134732, 0.268574018050262, 0.268735109343436, 0.268896651480838, 0.269058640757555, 0.269221073509133, 0.269221073509133, 0.269221073509133, 0.269660825841571, 0.270103700845149, 0.270549631043983, 0.270773721278562, 0.270998550867879, 0.271450396577676, 0.271905106194301, 0.272362619431318, 0.272362619431317, 0.272822877630747, 0.273285823701994, 0.273751402063673, 0.273985161383223, 0.274219558588186, 0.274690240548891, 0.275163396569704, 0.275638976577024, 0.275638976577024, 0.276116931753827, 0.27659721449583, 0.277079778369593, 0.277321901530699, 0.277564578072484, 0.278051569394377, 0.278540709181014, 0.279031955298933, 0.279031955298933, 0.279525266601889, 0.280020602898683, 0.280517924922332, 0.280767318556213, 0.281017194300518, 0.28151837352724, 0.282021425935624, 0.28252631567182, 0.28252631567182, 0.283033007669943, 0.283541467628005, 0.284051661984794, 0.284307399271899, 0.284563557897641, 0.285077123221058, 0.285592326486183, 0.286109136881007, 0.286109136881007, 0.286627524231343, 0.287147458982512, 0.287668912181694, 0.287930199311909, 0.288191855460934, 0.288716261020771, 0.289242101614449, 0.289769350532697, 0.289769350532697, 0.290297981589052, 0.290827969105702, 0.291359287899814, 0.291625438788374, 0.291891913270348, 0.292425820985324, 0.292960987269514, 0.293497388792568, 0.293497388792568, 0.293497388792568, 0.295384194836048, 0.296234575123112, 0.297284914683972, 0.299000110613969, 0.29919866415428, 0.301124612286544, 0.301791417693112, 0.303061977024441, 0.304606136177744, 0.305010021346535, 0.30696804978961, 0.30744209886063, 0.30893540531677, 0.31029731057822, 0.310911466489255, 0.312895644906541, 0.313169930170056, 0.314887382884064, 0.316058254853481, 0.316886151341973, 0.31889144788173, 0.318960706627375, 0.320902795030357, 0.320902795030357, 0.32187582038949, 0.322054677164543, 0.323208304819342, 0.324363598745772, 0.324802233508199, 0.325520481859477, 0.326678879164688, 0.327738676634559, 0.327838717681705, 0.328999926377672, 0.330162436100489, 0.330683965576837, 0.331326179515658, 0.332491091045929, 0.333636994089036, 0.333657106813568, 0.334824164585128, 0.335992203718576, 0.336596727448896, 0.337161165112649, 0.338330991158342, 0.339501625692387, 0.339562196720429, 0.340673013952656, 0.341845102535351, 0.342532493612219, 0.343017839353934, 0.344191173599672, 0.345365055703741, 0.345506765855998, 0.346539437300799, 0.347714271193974, 0.348484213041172, 0.348889511321174, 0.350065112722681, 0.351241031509952, 0.351464082850154, 0.352417224835577, 0.353593650864339, 0.354445667647157, 0.354770268745328, 0.355947038585055, 0.357123921421526, 0.357428301379611, 0.358300879199239, 0.35947787474504, 0.360411356756854, 0.360654871744832, 0.361831834721075, 0.363008729011056, 0.363394242675464, 0.364185520745887, 0.36536217683021, 0.366376401864514, 0.366538664922575, 0.367714953416453, 0.368891011421877, 0.36935730872746, 0.370066808747673, 0.371242315884251, 0.372336467360264, 0.372417503986955, 0.373592344859922, 0.374766810940453, 0.375313409727833, 0.375940875283867, 0.37711451154881, 0.378287693983021, 0.378287693983021]
	# S_1_P3_DHV =  [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0.357853665490959, 0.353828643198873, 0.350973390116319, 0.348696917050134, 0.346782232090199, 0.34512081599012, 0.343649206311263, 0.342326512317965, 0.341124574348701, 0.340800350152404, 0.340023032392881, 0.339006607723555, 0.338063490673598, 0.337184333376976, 0.336361588163858, 0.335589058607037, 0.334861584741025, 0.334174816574448, 0.333525047141211, 0.333081795161268, 0.332909088691435, 0.332324174806252, 0.331767888422041, 0.331238102827166, 0.330732935415622, 0.330250710453323, 0.329789928817276, 0.329349243181093, 0.328927437497306, 0.328523409902258, 0.328467083372334, 0.328136158371702, 0.327764768605612, 0.327408403733545, 0.327066295517374, 0.326737736794821, 0.326422074954522, 0.326118706279039, 0.325827071016412, 0.325546649057089, 0.325391254781664, 0.325276956216157, 0.325017540767131, 0.324767980626598, 0.324527880673531, 0.324296870436288, 0.324074602016869, 0.32386074823353, 0.323655000954388, 0.32345706959667, 0.323266679783735, 0.323266679783735, 0.323266679783735, 0.322789341238585, 0.322360301658311, 0.321975564146745, 0.321798698470641, 0.321631630884231, 0.321325421049929, 0.321054205230337, 0.3208155524506, 0.3208155524506, 0.320607286989288, 0.320427452861318, 0.320274284372987, 0.320207193861723, 0.320146181530678, 0.320041689363131, 0.319959480424614, 0.319898339902807, 0.319898339902807, 0.319857152874396, 0.319834892792086, 0.31983061476499, 0.319834942217659, 0.319843441825086, 0.319872563266003, 0.319917225143413, 0.319976727208811, 0.319976727208811, 0.320050415178559, 0.320137678366884, 0.320237945383117, 0.320292786668601, 0.320350680736349, 0.320475381804877, 0.320611576123899, 0.320758818951738, 0.320758818951738, 0.320916691080618, 0.321084796862757, 0.321262762426594, 0.321355331345533, 0.321450234061353, 0.321646876751043, 0.321852372841419, 0.322066420825558, 0.322066420825558, 0.322288734235459, 0.322519040628647, 0.322757080660069, 0.322878923115147, 0.323002607230718, 0.323255384705432, 0.323515188193141, 0.323781802883641, 0.323781802883641, 0.324055023435568, 0.324334653410871, 0.32462050475156, 0.324765704147108, 0.324912397294958, 0.325210158324063, 0.32551362214999, 0.325822629723758, 0.325822629723758, 0.325822629723758, 0.326945660394133, 0.327468925744823, 0.328128785763402, 0.329236271689032, 0.329366737395422, 0.330654869964424, 0.331109928360737, 0.331989063898841, 0.33307752512464, 0.333365646607911, 0.334781328156563, 0.335128564524692, 0.336233148294912, 0.337254051817794, 0.3377184324992, 0.33923475522956, 0.339446213156415, 0.340779909015835, 0.341698277932553, 0.342351878286566, 0.343948817086057, 0.344004308379329, 0.345569030000035, 0.345569030000035, 0.346359064529814, 0.346504709610835, 0.347447203601973, 0.348396250639769, 0.34875789600579, 0.349351600675059, 0.350313014284768, 0.351196654428026, 0.351280262062069, 0.352253124050838, 0.353231389220552, 0.353671621862128, 0.354214854978164, 0.355203326713842, 0.356179451867529, 0.356196617377747, 0.35719454708532, 0.358196942748761, 0.358717120549378, 0.359203637732616, 0.360214471531592, 0.361229289468855, 0.361281885621545, 0.362247942413261, 0.363270286514089, 0.363871251939376, 0.364296182951968, 0.36532549770481, 0.366358101327661, 0.366482942298013, 0.367393868745474, 0.368432679057888, 0.369114872547251, 0.36947441535517, 0.37051896454456, 0.371566217186294, 0.371765130268843, 0.372616067338675, 0.373668412411561, 0.374431956412461, 0.374723153027744, 0.37578019289168, 0.37683943866513, 0.377113729403422, 0.377900799849233, 0.378964188672638, 0.379808951326901, 0.380029519985309, 0.38109671115764, 0.382165681984585, 0.382516235865702, 0.38323635459447, 0.384308653362235, 0.38523429772623, 0.385382504826823, 0.386457837612495, 0.387534582353823, 0.387961943333343, 0.38861267162417, 0.389692039867448, 0.390698062611901, 0.390772623332973, 0.391854360013251, 0.392937189584524, 0.39344162170288, 0.394021053349939, 0.395105894185179, 0.396191656486459, 0.396191656486459]
	
	# T0 =  [270.0, 270.6896551724138, 271.37931034482756, 272.0689655172414, 272.7586206896552, 273.44827586206895, 274.13793103448273, 274.82758620689657, 275.51724137931035, 276.2068965517241, 276.8965517241379, 277.58620689655174, 278.2758620689655, 278.9655172413793, 279.6551724137931, 280.3448275862069, 281.0344827586207, 281.7241379310345, 282.41379310344826, 283.1034482758621, 283.7931034482759, 284.48275862068965, 285.17241379310343, 285.86206896551727, 286.55172413793105, 287.2413793103448, 287.9310344827586, 288.62068965517244, 289.3103448275862, 290.0, 290.0, 290.33898305084745, 290.6779661016949, 291.0169491525424, 291.35593220338984, 291.6949152542373, 292.03389830508473, 292.3728813559322, 292.7118644067797, 293.0508474576271, 293.3898305084746, 293.728813559322, 294.06779661016947, 294.40677966101697, 294.7457627118644, 295.08474576271186, 295.4237288135593, 295.76271186440675, 296.10169491525426, 296.4406779661017, 296.77966101694915, 297.1186440677966, 297.45762711864404, 297.79661016949154, 298.135593220339, 298.47457627118644, 298.8135593220339, 299.1525423728814, 299.49152542372883, 299.8305084745763, 300.1694915254237, 300.50847457627117, 300.8474576271186, 301.1864406779661, 301.52542372881356, 301.864406779661, 302.20338983050846, 302.54237288135596, 302.8813559322034, 303.22033898305085, 303.5593220338983, 303.89830508474574, 304.23728813559325, 304.5762711864407, 304.91525423728814, 305.2542372881356, 305.59322033898303, 305.93220338983053, 306.271186440678, 306.6101694915254, 306.9491525423729, 307.2881355932203, 307.6271186440678, 307.96610169491527, 308.3050847457627, 308.64406779661016, 308.9830508474576, 309.3220338983051, 309.66101694915255, 310.0, 310.0, 310.5263157894737, 311.05263157894734, 311.57894736842104, 312.10526315789474, 312.63157894736844, 313.1578947368421, 313.6842105263158, 314.2105263157895, 314.7368421052632, 315.2631578947368, 315.7894736842105, 316.3157894736842, 316.8421052631579, 317.36842105263156, 317.89473684210526, 318.42105263157896, 318.94736842105266, 319.4736842105263, 320.0, 320.0, 322.10526315789474, 324.2105263157895, 326.3157894736842, 328.42105263157896, 330.5263157894737, 332.63157894736844, 334.7368421052632, 336.8421052631579, 338.9473684210526, 341.05263157894734, 343.1578947368421, 345.2631578947368, 347.36842105263156, 349.4736842105263, 351.57894736842104, 353.6842105263158, 355.7894736842105, 357.89473684210526, 360.0]
	# S_1_P1_DHV =  [0.201767465038180, 0.202829472626453, 0.203892339321574, 0.204956030279786, 0.206020511213571, 0.207085748381164, 0.208151708576319, 0.209218359118349, 0.210285667842408, 0.211353603090025, 0.212422133699864, 0.213491228998725, 0.214560858792754, 0.215630993358877, 0.216701603436440, 0.217772660219045, 0.218844135346589, 0.219916000897487, 0.220988229381084, 0.222060793730239, 0.223133667294083, 0.224206823830953, 0.225280237501474, 0.226353882861812, 0.227427734857071, 0.228501768814846, 0.229575960438911, 0.230650285803057, 0.231724721345061, 0.232799243860783, 0.232799243860783, 0.233327423906517, 0.233855616755776, 0.234383819745780, 0.234912030233666, 0.235440245596330, 0.235968463230157, 0.236496680551463, 0.237024894995148, 0.237553104015751, 0.238081305086681, 0.238609495700218, 0.239137673367350, 0.239665835617636, 0.240193979999062, 0.240722104077900, 0.241250205438571, 0.241778281683505, 0.242306330433007, 0.242834349325117, 0.243362336015481, 0.243890288177218, 0.244418203500788, 0.244946079693861, 0.245473914481191, 0.246001705604485, 0.246529450822283, 0.247057147909827, 0.247584794658940, 0.248112388877907, 0.248639928391346, 0.249167411040095, 0.249694834681091, 0.250222197187251, 0.250749496447354, 0.251276730365933, 0.251803896863149, 0.252330993874690, 0.252858019351647, 0.253384971260158, 0.253911847582563, 0.254438646314755, 0.254965365468615, 0.255492003070630, 0.256018557162045, 0.256545025798756, 0.257071407051206, 0.257597699004280, 0.258123899757207, 0.258650007423452, 0.259176020130589, 0.259701936020366, 0.260227753248265, 0.260753469983754, 0.261279084410005, 0.261804594723844, 0.262329999135625, 0.262855295869269, 0.263380483161900, 0.263905559264025, 0.263905559264025, 0.264720584590672, 0.265535331255777, 0.266349792898119, 0.267163963230931, 0.267977836041113, 0.268791405188446, 0.269604664604823, 0.270417608293484, 0.271230230328270, 0.272042524852874, 0.272854486080116, 0.273666108291213, 0.274477385835073, 0.275288313127582, 0.276098884650914, 0.276909094952842, 0.277718938646061, 0.278528410407515, 0.279337504977738, 0.279337504977738, 0.282570008346249, 0.285796071993438, 0.289015388466197, 0.292227665010146, 0.295432622986832, 0.298629997317505, 0.301819535951867, 0.305000999360279, 0.308174160048194, 0.311338802091126, 0.314494720689691, 0.317641721742923, 0.320779621439302, 0.323908245864411, 0.327027430624377, 0.330137020484304, 0.333236869020942, 0.336326838288891, 0.339406798499705]
	# S_1_P2_DHV =  [0.261884259438146, 0.262410794214241, 0.262945836254052, 0.263489203485482, 0.264040718232507, 0.264600206958081, 0.265167500164207, 0.265742432266824, 0.266324841475792, 0.266914569679727, 0.267511462313935, 0.268115368340955, 0.268726140017176, 0.269343632883141, 0.269967705646904, 0.270598220093280, 0.271235040996510, 0.271878036036181, 0.272527075716245, 0.273182033286996, 0.273842784669871, 0.274509208384950, 0.275181185481024, 0.275858599468126, 0.276541336252418, 0.277229284073319, 0.277922333442789, 0.278620377086676, 0.279323309888024, 0.280031028832279, 0.280031028832279, 0.280380614286197, 0.280731319823993, 0.281083133749207, 0.281436044479154, 0.281790040543533, 0.282145110583057, 0.282501243348116, 0.282858427697445, 0.283216652596826, 0.283575907117800, 0.283936180436410, 0.284297461831953, 0.284659740685757, 0.285023006479977, 0.285387248796407, 0.285752457315308, 0.286118621814264, 0.286485732167042, 0.286853778342477, 0.287222750403372, 0.287592638505415, 0.287963432896114, 0.288335123913738, 0.288707701986289, 0.289081157630477, 0.289455481450714, 0.289830664138125, 0.290206696469569, 0.290583569306676, 0.290961273594901, 0.291339800362586, 0.291719140720039, 0.292099285858627, 0.292480227049877, 0.292861955644597, 0.293244463071999, 0.293627740838849, 0.294011780528615, 0.294396573800632, 0.294782112389283, 0.295168388102354, 0.295555392824383, 0.295943118507576, 0.296331557179322, 0.296720700937278, 0.297110541949442, 0.297501072453404, 0.297892284755610, 0.298284171230634, 0.298676724320462, 0.299069936533783, 0.299463800445288, 0.299858308694987, 0.300253453987522, 0.300649229091500, 0.301045626838830, 0.301442640124069, 0.301840261903776, 0.302238485195876, 0.302238485195876, 0.302857955017899, 0.303478832597380, 0.304101092658865, 0.304724710265364, 0.305349660813063, 0.305975920026137, 0.306603463951662, 0.307232268954631, 0.307862311713061, 0.308493569213198, 0.309126018744810, 0.309759637896577, 0.310394404551556, 0.311030296882744, 0.311667293348717, 0.312305372689356, 0.312944513921643, 0.313584696335548, 0.314225899489979, 0.314225899489979, 0.316800519883762, 0.319389904125383, 0.321992864872876, 0.324608274295334, 0.327235060764384, 0.329872205768501, 0.332518741031852, 0.335173745821177, 0.337836344425790, 0.340505703797201, 0.343181031336101, 0.345861572815600, 0.348546610430563, 0.351235460963833, 0.353927474060875, 0.356622030605267, 0.359318541187677, 0.362016444662234, 0.364715206784091]
	# S_1_P3_DHV =  [0.271269876485035, 0.272444613613497, 0.273617943829939, 0.274789833366556, 0.275960248280739, 0.277129154435239, 0.278296517477076, 0.279462302815042, 0.280626475595716, 0.281789000677839, 0.282949842604902, 0.284108965575788, 0.285266333413272, 0.286421909530190, 0.287575656893029, 0.288727537982688, 0.289877514752117, 0.291025548580497, 0.292171600223601, 0.293315629759897, 0.294457596531926, 0.295597459082381, 0.296735175084292, 0.297870701261021, 0.299003993320016, 0.300135005825046, 0.301263692129650, 0.302390004246626, 0.303513892726350, 0.304635306517340, 0.304635306517340, 0.305185586776725, 0.305735249991033, 0.306284289604154, 0.306832698936108, 0.307380471176904, 0.307927599380004, 0.308474076455380, 0.309019895162105, 0.309565048100469, 0.310109527703546, 0.310653326228189, 0.311196435745394, 0.311738848129965, 0.312280555049432, 0.312821547952137, 0.313361818054418, 0.313901356326793, 0.314440153479039, 0.314978199944071, 0.315515485860472, 0.316052001053531, 0.316587735014643, 0.317122676878852, 0.317656815400349, 0.318190138925664, 0.318722635364265, 0.319254292155839, 0.319785096236778, 0.320315033996661, 0.320844091238857, 0.321372253129999, 0.321899504146320, 0.322425828013686, 0.322951207638912, 0.323475625034221, 0.323999061230325, 0.324521496178228, 0.325042908636988, 0.325563276044884, 0.326082574370743, 0.326600777941313, 0.327117859239551, 0.329117040500021, 0.329482870277774, 0.329845110953650, 0.330205605362568, 0.330564808340542, 0.330923015025195, 0.331280436357041, 0.331637232319145, 0.331993192188600, 0.332349429801170, 0.332705019356848, 0.333060101203031, 0.333415292544708, 0.333770358593809, 0.334125346499045, 0.334480297796866, 0.334835249321244, 0.334835249321244, 0.335386434676827, 0.335937806035250, 0.336489454658165, 0.337041458802461, 0.337593885975949, 0.338146794684975, 0.338700235812884, 0.339254253724367, 0.339808887162358, 0.340364169985279, 0.340920131779580, 0.341476798373578, 0.342034192272233, 0.342592333027900, 0.343151237558739, 0.343710920423943, 0.344271394063036, 0.344832669005082, 0.345394754052475, 0.345394754052475, 0.347651319498688, 0.349921198016473, 0.352204395907410, 0.354500737691956, 0.356809918027055, 0.359131538352576, 0.361465133434784, 0.363810190984926, 0.366166166391263, 0.368532493918732, 0.370908595303028, 0.373293886389662, 0.375687782284800, 0.378089701359228, 0.380499068359196, 0.382915316815590, 0.385337890897682, 0.387766246824474, 0.390199853921859]

	# T0 =  [270.0, 270.6896551724138, 271.37931034482756, 272.0689655172414, 272.7586206896552, 273.44827586206895, 274.13793103448273, 274.82758620689657, 275.51724137931035, 276.2068965517241, 276.8965517241379, 277.58620689655174, 278.2758620689655, 278.9655172413793, 279.6551724137931, 280.3448275862069, 281.0344827586207, 281.7241379310345, 282.41379310344826, 283.1034482758621, 283.7931034482759, 284.48275862068965, 285.17241379310343, 285.86206896551727, 286.55172413793105, 287.2413793103448, 287.9310344827586, 288.62068965517244, 289.3103448275862, 290.0, 290.0, 290.6896551724138, 291.37931034482756, 292.0689655172414, 292.7586206896552, 293.44827586206895, 294.13793103448273, 294.82758620689657, 295.51724137931035, 296.2068965517241, 296.8965517241379, 297.58620689655174, 298.2758620689655, 298.9655172413793, 299.6551724137931, 300.3448275862069, 301.0344827586207, 301.7241379310345, 302.41379310344826, 303.1034482758621, 303.7931034482759, 304.48275862068965, 305.17241379310343, 305.86206896551727, 306.55172413793105, 307.2413793103448, 307.9310344827586, 308.62068965517244, 309.3103448275862, 310.0, 310.0, 310.3448275862069, 310.6896551724138, 311.0344827586207, 311.37931034482756, 311.7241379310345, 312.0689655172414, 312.41379310344826, 312.7586206896552, 313.1034482758621, 313.44827586206895, 313.7931034482759, 314.13793103448273, 314.48275862068965, 314.82758620689657, 315.17241379310343, 315.51724137931035, 315.86206896551727, 316.2068965517241, 316.55172413793105, 316.8965517241379, 317.2413793103448, 317.58620689655174, 317.9310344827586, 318.2758620689655, 318.62068965517244, 318.9655172413793, 319.3103448275862, 319.6551724137931, 320.0, 320.0, 321.37931034482756, 322.7586206896552, 324.13793103448273, 325.51724137931035, 326.8965517241379, 328.2758620689655, 329.6551724137931, 331.0344827586207, 332.41379310344826, 333.7931034482759, 335.17241379310343, 336.55172413793105, 337.9310344827586, 339.3103448275862, 340.6896551724138, 342.0689655172414, 343.44827586206895, 344.82758620689657, 346.2068965517241, 347.58620689655174, 348.9655172413793, 350.3448275862069, 351.7241379310345, 353.1034482758621, 354.48275862068965, 355.86206896551727, 357.2413793103448, 358.62068965517244, 360.0]
	# S_1_P1_DHV =  [0.201767465038180, 0.202829472626453, 0.203892339321574, 0.204956030279786, 0.206020511213571, 0.207085748381164, 0.208151708576319, 0.209218359118349, 0.210285667842408, 0.211353603090025, 0.212422133699864, 0.213491228998725, 0.214560858792754, 0.215630993358877, 0.216701603436440, 0.217772660219045, 0.218844135346589, 0.219916000897487, 0.220988229381084, 0.222060793730239, 0.223133667294083, 0.224206823830953, 0.225280237501474, 0.226353882861812, 0.227427734857071, 0.228501768814846, 0.229575960438911, 0.230650285803057, 0.231724721345061, 0.232799243860783, 0.232799243860783, 0.233873830498399, 0.234948458752754, 0.236023106459840, 0.237097751791380, 0.238172373249546, 0.239246949661768, 0.240321460175663, 0.241395884254069, 0.242470201670170, 0.243544392502737, 0.244618437131450, 0.245692316232323, 0.246766010773220, 0.247839502009453, 0.248912771479477, 0.249985801000661, 0.251058572665142, 0.252131068835766, 0.253203272141483, 0.254275165476505, 0.255346731990338, 0.256417955090146, 0.257488818433995, 0.258559305927834, 0.259629401721940, 0.260699090207421, 0.261768356012783, 0.262837184000565, 0.263905559264025, 0.263905559264025, 0.264439572528064, 0.264973467123890, 0.265507241251614, 0.266040893125165, 0.266574420972196, 0.267107823033991, 0.267641097565364, 0.268174242834568, 0.268707257123203, 0.269240138726121, 0.269772885951334, 0.270305496647521, 0.270837970565953, 0.271370304636373, 0.271902497690937, 0.272434548102110, 0.272966454254986, 0.273498214547194, 0.274029827388820, 0.274561291202317, 0.275092604422423, 0.275623765496075, 0.276154772882328, 0.276685625052273, 0.277216320488953, 0.277746857687283, 0.278277235153971, 0.278807451407439, 0.279337504977738, 0.279337504977738, 0.281456063431125, 0.283571904330027, 0.285684939015865, 0.287795081614961, 0.289902248965264, 0.292006360545336, 0.294107338405481, 0.296205107100941, 0.298299593627077, 0.300390727356448, 0.302478439977366, 0.304562665436321, 0.306643339876902, 0.308720401587169, 0.310793790943454, 0.312863450357924, 0.314929324226974, 0.316991358881181, 0.319049502536575, 0.321103705247227, 0.323153918859085, 0.325200096965040, 0.327242194861168, 0.329280169504104, 0.331313979469529, 0.333343584911718, 0.335368947524122, 0.337390030500945, 0.339406798499705]
	# S_1_P2_DHV =  [0.261884259438146, 0.262410794214241, 0.262945836254052, 0.263489203485482, 0.264040718232507, 0.264600206958081, 0.265167500164207, 0.265742432266824, 0.266324841475792, 0.266914569679727, 0.267511462313935, 0.268115368340955, 0.268726140017176, 0.269343632883141, 0.269967705646904, 0.270598220093280, 0.271235040996510, 0.271878036036181, 0.272527075716245, 0.273182033286996, 0.273842784669871, 0.274509208384950, 0.275181185481024, 0.275858599468126, 0.276541336252418, 0.277229284073319, 0.277922333442789, 0.278620377086676, 0.279323309888024, 0.280031028832279, 0.280031028832279, 0.280743432954305, 0.281460423287123, 0.282181902812328, 0.282907776412093, 0.283637950822703, 0.284372334589564, 0.285110838023622, 0.285853373159134, 0.286599853712745, 0.287350195043817, 0.288104314115963, 0.288862129459737, 0.289623561136439, 0.290388530702995, 0.291156961177870, 0.291928777007971, 0.292703904036513, 0.293482269471805, 0.294263801856933, 0.295048431040292, 0.295836088146954, 0.296626705550830, 0.297420216847602, 0.298216556828409, 0.299015661454235, 0.299817467831006, 0.300621914185352, 0.301428939841012, 0.302238485195876, 0.302238485195876, 0.302644184423047, 0.303050491699621, 0.303457399867577, 0.303864901831942, 0.304272990560138, 0.304681659081343, 0.305090900485858, 0.305500707924480, 0.305911074607888, 0.306321993806032, 0.306733458847533, 0.307145463119088, 0.307558000064885, 0.307971063186026, 0.308384646039951, 0.308798742239880, 0.309213345454250, 0.309628449406169, 0.310044047872872, 0.310460134685181, 0.310876703726981, 0.311293748934689, 0.311711264296745, 0.312129243853096, 0.312547681694693, 0.312966571962994, 0.313385908849469, 0.313805686595115, 0.314225899489979, 0.314225899489979, 0.315910990052707, 0.317602598640747, 0.319300382363663, 0.321004009738070, 0.322713160263400, 0.324427524016935, 0.326146801267023, 0.327870702103493, 0.329598946084312, 0.331331261897621, 0.333067387038336, 0.334807067498536, 0.336550057470925, 0.338296119064706, 0.340045022033217, 0.341796543512747, 0.343550467771976, 0.345306585971511, 0.347064695933022, 0.348824601917524, 0.350586114412357, 0.352349049926439, 0.354113230793504, 0.355878484982710, 0.357644645916591, 0.359411552295789, 0.361179047930353, 0.362946981577290, 0.364715206784091]
	# S_1_P3_DHV =  [0.271192611200941, 0.272369502930067, 0.273545049193161, 0.274719217994749, 0.275891977247788, 0.277063294759242, 0.278233138214829, 0.279401475162886, 0.280568272997268, 0.281733498939228, 0.282897120018167, 0.284059103051191, 0.285219414621347, 0.286378021054450, 0.287534888394359, 0.288689982376579, 0.289843268400021, 0.290994711496771, 0.292144276299654, 0.293291927007410, 0.294437627347216, 0.295581340534307, 0.296723029228396, 0.297862655486545, 0.299000180712113, 0.300135565599346, 0.301268770073114, 0.302399753223238, 0.303528473232759, 0.304654887299414, 0.304654887299414, 0.305778951549460, 0.306900620942884, 0.308019849168825, 0.309136588529914, 0.310250789813945, 0.311362402151073, 0.312471372854363, 0.313577647241130, 0.314681168432026, 0.315781877124179, 0.316879711333977, 0.317974606104094, 0.319066493168178, 0.320155300565075, 0.321240952192480, 0.322323367287360, 0.323402459817111, 0.324478137760940, 0.325550302254951, 0.326618846566175, 0.327683654849415, 0.328744600624706, 0.329801544890096, 0.330854333750547, 0.331902795392786, 0.332946736157036, 0.333985935330729, 0.335020138080978, 0.336049045582444, 0.336049045582444, 0.336561404820044, 0.337072300741714, 0.337581676090390, 0.338089466648410, 0.338595599735109, 0.339099992216420, 0.339602547803676, 0.340103153279573, 0.340601673038558, 0.341097940835099, 0.341820635080433, 0.342222064769821, 0.342618242834770, 0.343011442793556, 0.343402487679004, 0.343791848934824, 0.344179841465474, 0.344566694046704, 0.344952582042570, 0.345337644991397, 0.345721997021786, 0.346105733468883, 0.346488935306098, 0.346871672240143, 0.347254004946796, 0.347635986731659, 0.348017664793105, 0.348399081202117, 0.348780273675744, 0.348780273675744, 0.350303437825856, 0.351825242393504, 0.353346910797906, 0.354869330263970, 0.356393157040431, 0.357918881276889, 0.359446869547955, 0.360977394066952, 0.362510653516254, 0.364046788354250, 0.365585892346158, 0.367128021432128, 0.368673200667925, 0.370221429738668, 0.371772687395410, 0.373326935064607, 0.374884119812924, 0.376444176802872, 0.378007031341537, 0.379572600600679, 0.381140795068908, 0.382711519783558, 0.384284675380047, 0.385860158988943, 0.387437865005189, 0.389017685749371, 0.390599512037354, 0.392183233671791, 0.393768739866705]

	if Isobars:

		if Plot_Phi:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			if P1!=0.0:
				plt.plot(T0,phip_P1_DHV,'r',ls=ls1,label='phi_p_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phip_P2_DHV,'r',ls=ls2,label='phi_p_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phip_P3_DHV,'r',ls=ls3,label='phi_p_{} MPa'.format(P3),lw=linewidth)
			
			if P1!=0.0:
				plt.plot(T0,phis_P1_DHV,'m',ls=ls1,label='phi_s_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phis_P2_DHV,'m',ls=ls2,label='phi_s_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phis_P3_DHV,'m',ls=ls3,label='phi_s_{} MPa'.format(P3),lw=linewidth)
			
			if P1!=0.0:
				plt.plot(T0,Rtilde_P1_DHV,'b',ls=ls1,label='Rtilde_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Rtilde_P2_DHV,'b',ls=ls2,label='Rtilde_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Rtilde_P3_DHV,'b',ls=ls3,label='Rtilde_{} MPa'.format(P3),lw=linewidth)
			
			if P1!=0.0:
				plt.plot(T0,phip0_P1_DHV,'k',ls=ls1,label='phi_p0_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,phip0_P2_DHV,'k',ls=ls2,label='phi_p0_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,phip0_P3_DHV,'k',ls=ls3,label='phi_p0_{} MPa'.format(P3),lw=linewidth)
			
			if P1!=0.0:
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
			
			if P1!=0.0:
				plt.plot(T0,S_1_P1_DHV,'r',ls=ls2,label='Present Theory at {} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,S_1_P2_DHV,'b',ls=ls1,label='{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,S_1_P3_DHV,'g',ls=ls3,label='{} MPa'.format(P3),lw=linewidth)

			if Plot_S2:
				if P1!=0.0:
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
			figS.savefig('./'+output_folder+r'\PS_CO2_Self_Grassia_02kilo_POST_THESIS_Paper4_11_12_Entropy_Low_phis0'+img_extension,dpi=240)


if Condo_Original:
	if Isotherms:
		if Plot_Phi:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			if T1!=0.0:
				plt.plot(P0,Rtildem_T1,'r',ls=ls1,label='Rtildem_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Rtildem_T2,'r',ls=ls2,label='Rtildem_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Rtildem_T3,'r',ls=ls3,label='Rtildem_{} K'.format(T3),lw=linewidth)
			
		
			if T1!=0.0:
				plt.plot(P0,cphis_T1,'m',ls=ls1,label='cphi_s_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,cphis_T2,'m',ls=ls2,label='cphi_s_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,cphis_T3,'m',ls=ls3,label='cphi_s_{} K'.format(T3),lw=linewidth)
			
			
			if T1!=0.0:
				plt.plot(P0,Rtildep0_T1,'b',ls=ls1,label='Rtildep0_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Rtildep0_T2,'b',ls=ls2,label='Rtildep0_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Rtildep0_T3,'b',ls=ls3,label='Rtildep0_{} K'.format(T3),lw=linewidth)
			

			if T1!=0.0:
				plt.plot(P0,Rtildes0_T1,'k',ls=ls1,label='Rtildes0_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,Rtildes0_T2,'k',ls=ls2,label='Rtildes0_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,Rtildes0_T3,'k',ls=ls3,label='Rtildes0_{} K'.format(T3),lw=linewidth)
			

			# if T1!=0.0:
			# 	plt.plot(P0,phis0_T1,'y',ls=ls1,label='phi_s0_{} K'.format(T1),lw=linewidth)
			# if T2!=0.0:
			# 	plt.plot(P0,phis0_T2,'y',ls=ls2,label='phi_s0_{} K'.format(T2),lw=linewidth)
			# if T3!=0.0:
			# 	plt.plot(P0,phis0_T3,'y',ls=ls3,label='phi_s0_{} K'.format(T3),lw=linewidth)

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
				plt.plot(P0,S_1_T1,'r',ls=ls2,label='S_1_{} K'.format(T1),lw=linewidth)
			if T2!=0.0:
				plt.plot(P0,S_1_T2,'b',ls=ls1,label='S_1_{} K'.format(T2),lw=linewidth)
			if T3!=0.0:
				plt.plot(P0,S_1_T3,'g',ls=ls3,label='S_1_{} K'.format(T3),lw=linewidth)

			if Plot_S2:
				if T1!=0.0:
					plt.plot(P0,S_2_T1,'m',ls=ls1,label='S_2_{} K'.format(T1),lw=linewidth)
				if T2!=0.0:
					plt.plot(P0,S_2_T2,'m',ls=ls2,label='S_2_{} K'.format(T2),lw=linewidth)
				if T3!=0.0:
					plt.plot(P0,S_2_T3,'m',ls=ls3,label='S_2_{} K'.format(T3),lw=linewidth)

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
	# S_1_P1 =  [0.287361847792063, 0.291977922320856, 0.296529918157662, 0.301007532620532, 0.305400599881289, 0.309699376537802, 0.313894818619888, 0.317978813960745, 0.321944333142034, 0.325785467296709, 0.329497329826039, 0.333075806785722, 0.336517140069245, 0.339817307683451, 0.342971104478470, 0.345970673236043, 0.348802759621146, 0.351442311597064, 0.353830020283959, 0.354955038334893, 0.235996868936240, 0.237259747553113, 0.240358026746013, 0.243456167291127, 0.245029036669523, 0.245145336632522, 0.246553748583989, 0.249650365907212, 0.252745629835920, 0.255411570700446, 0.255527691192725, 0.255839165673704, 0.258930612916777, 0.262019624744252, 0.265105867532641, 0.268189020392870, 0.271268774728241, 0.274344833811948, 0.277416912382827, 0.280484736258203, 0.283548041962723, 0.286606576372236, 0.289660096371782, 0.292708368526901, 0.295751168767483, 0.298788282083481, 0.301819502231834, 0.304844631454020, 0.307863480203684, 0.310875866883849, 0.313881617593231, 0.316880565881230, 0.319872552511194, 0.322857425231579, 0.325835038554655, 0.328805253542435, 0.331767937599515, 0.334722964272544, 0.337670213056056, 0.340609569204406, 0.343540923549576, 0.346464172324634, 0.349379216992622, 0.352285964080678, 0.355184325019218, 0.358074215985969]
	# S_1_P2 =  [0.291022057782900, 0.295951958184037, 0.300827160820633, 0.305632488542333, 0.310351893051314, 0.314969061154339, 0.319468210971551, 0.323834995319327, 0.328057371303902, 0.332126266755621, 0.336035898049091, 0.339783669013325, 0.343369677231056, 0.346795929576319, 0.350065393301933, 0.353180977390338, 0.356144461918853, 0.358955271644787, 0.361608772040304, 0.362996004843946, 0.363102108978563, 0.364093213531999, 0.366382444351142, 0.368409206845696, 0.369254443642157, 0.280974428245624, 0.282017155010234, 0.284337769022368, 0.286693532391806, 0.288749542801756, 0.288839638822677, 0.289081525161202, 0.291499089361689, 0.293943797306727, 0.296413424706242, 0.298905927735525, 0.301419423370539, 0.303952172438047, 0.306502564935414, 0.309069107258429, 0.311650411041466, 0.314245183366871, 0.316852218142476, 0.319470388480101, 0.322098639935410, 0.324735984491884, 0.327381495190093, 0.330034301318572, 0.332693584095140, 0.335358572777958, 0.338028541154270, 0.340702804362138, 0.343380716006572, 0.346061665536674, 0.348745075854830, 0.351430401132703, 0.354117124812011, 0.356804757770787, 0.359492836638208, 0.362180922243063, 0.364868598182742, 0.367555469501089, 0.370241161464831, 0.372925318427314, 0.375607602786093, 0.378287693983021]
	# S_1_P3 =  [0.296037932340702, 0.301581175172076, 0.307137419685592, 0.312690031292753, 0.318214659292667, 0.323675969040932, 0.329024064935090, 0.334192707237204, 0.339104074058606, 0.343685371594877, 0.347893976806592, 0.351733218294751, 0.355244307057159, 0.358483728851476, 0.361504663717883, 0.364349052959828, 0.367046848807795, 0.369617933419464, 0.372074297283704, 0.373377433782964, 0.373477827175489, 0.374421628253627, 0.376660085736335, 0.378784093336241, 0.379814955600515, 0.379889817146639, 0.380780609381485, 0.382623948273740, 0.384258436882642, 0.385349208975200, 0.323315782316191, 0.323435194381678, 0.324696610481343, 0.326083367334617, 0.327580249512193, 0.329174544899525, 0.330855509800069, 0.332613971554761, 0.334442028011165, 0.336332816619509, 0.338280334506667, 0.340279296487738, 0.342325021727640, 0.344413342327578, 0.346540528893240, 0.348703229401508, 0.350898418586876, 0.353123355726980, 0.355375549191849, 0.357652726483330, 0.359952808763922, 0.362273889082024, 0.364614213660276, 0.366972165737500, 0.369346251551560, 0.371735088126690, 0.374137392589390, 0.376551972785267, 0.378977719008094, 0.381413596683766, 0.383858639877400, 0.386311945512709, 0.388772668204453, 0.391240015663211, 0.393713244488567, 0.396191656486459]

	# T0 = [290.0, 290.6779661, 291.3559322, 292.03389831, 292.71186441, 293.38983051, 294.06779661, 294.74576271, 295.42372881, 296.10169492, 296.77966102, 297.45762712, 298.13559322, 298.81355932, 299.49152542, 299.69421925, 299.77421925, 300.16949153, 300.84745763, 301.52542373, 302.20338983, 302.88135593, 303.55932203, 304.23728814, 304.91525424, 305.59322034, 306.27118644, 306.94915254, 307.54260373, 307.61260373, 307.62711864, 308.30508475, 308.98305085, 309.66101695, 310.33898305, 311.01694915, 311.69491525, 312.37288136, 313.05084746, 313.72881356, 314.40677966, 315.08474576, 315.45711514, 315.52711514, 315.76271186, 316.44067797, 317.11864407, 317.79661017, 318.47457627, 319.15254237, 319.83050847, 320.50847458, 321.18644068, 321.86440678, 322.54237288, 323.22033898, 323.89830508, 324.57627119, 325.25423729, 325.93220339, 326.61016949, 327.28813559, 327.96610169, 328.6440678, 329.3220339, 330.0]
	# S_1_P1 =  [0.304456946618227, 0.305535439151210, 0.306610366445478, 0.307681572487271, 0.308748881756768, 0.309812094417380, 0.310870979703552, 0.311925266539211, 0.312974629697035, 0.314018668348726, 0.315056870618981, 0.316088549641070, 0.317112712398667, 0.318127727294431, 0.319129979723347, 0.319425265231469, 0.308433997032113, 0.308819780595928, 0.309484643880623, 0.310153451302620, 0.310826130050739, 0.311502608365221, 0.312182815528396, 0.312866681854758, 0.313554138680535, 0.314245118352825, 0.314939554218368, 0.315637380611998, 0.316250948281517, 0.316323487125867, 0.316338532844861, 0.317042947192401, 0.317750560882202, 0.318461312081684, 0.319175139885718, 0.319891984304171, 0.320611786249415, 0.321334487523820, 0.322060030807265, 0.322788359644665, 0.323519418433548, 0.324253152411690, 0.324657270658008, 0.324733326555533, 0.324989507644827, 0.325728431014442, 0.326469870205657, 0.327213773695222, 0.327960090739617, 0.328708771363274, 0.329459766346927, 0.330213027216077, 0.330968506229616, 0.331726156368562, 0.332485931324955, 0.333247785490884, 0.334011673947669, 0.334777552455181, 0.335545377441316, 0.336315105991617, 0.337086695839044, 0.337860105353893, 0.338635293533870, 0.339412219994307, 0.340190844958531, 0.340971129248381]
	# S_1_P2 =  [0.304604992900903, 0.305699988958015, 0.306792293327434, 0.307881839018114, 0.308968555442556, 0.310052367969980, 0.311133197403620, 0.312210959364869, 0.313285563561954, 0.314356912913956, 0.315424902491472, 0.316489418221828, 0.317550335287543, 0.318607516118541, 0.319660807836284, 0.319974933884319, 0.320098813458606, 0.320710038942611, 0.321755014941862, 0.322795512412578, 0.323831270747535, 0.324861980239833, 0.325887264144315, 0.326906650144002, 0.327919521534309, 0.328925024732612, 0.329921864507478, 0.330907704561730, 0.331756260944449, 0.328950917619284, 0.328965508260804, 0.329648130830616, 0.330332954889257, 0.331019990312008, 0.331709242197215, 0.332400711470861, 0.333094395402752, 0.333790288050117, 0.334488380641061, 0.335188661907786, 0.335891118377574, 0.336595734627996, 0.336983654225647, 0.337056649371084, 0.337302493511653, 0.338011376354816, 0.338722363133583, 0.339435432630585, 0.340150562574784, 0.340867729766499, 0.341586910189509, 0.342308079111767, 0.343031211176072, 0.343756280481851, 0.344483260659036, 0.345212124934913, 0.345942846194685, 0.346675397036410, 0.347409749820892, 0.348145876717038, 0.348883749743114, 0.349623340804304, 0.350364621726995, 0.351107564289850, 0.351852140252379, 0.352598321380823]
	# S_1_P3 =  [0.304654887299414, 0.305759919435512, 0.306862637959683, 0.307962998908640, 0.309060957028133, 0.310156465655678, 0.311249476590301, 0.312339939947410, 0.313427803996553, 0.314513014979379, 0.315595516904638, 0.316675251316362, 0.317752157030595, 0.318826169835022, 0.319897222144546, 0.320216852908515, 0.320342931028261, 0.320965242604252, 0.322030155629043, 0.323091880866508, 0.324150332565910, 0.325205418831352, 0.326257040730580, 0.327305091221908, 0.328349453849119, 0.329390001136410, 0.330426592589575, 0.331459072171344, 0.332359331425945, 0.332465298188345, 0.332487265060716, 0.333510973415233, 0.334529970708341, 0.335543993967305, 0.336552732803106, 0.337555813316722, 0.338552773359378, 0.339543022136488, 0.340525768660621, 0.341499879367877, 0.342463537189853, 0.343413065445634, 0.343924283651724, 0.343802976464552, 0.344068172783527, 0.344828409215007, 0.345585268912219, 0.346339610450463, 0.347092045343891, 0.347843033460844, 0.348592933651104, 0.349342033475209, 0.350090567977070, 0.350838732206380, 0.351586689938952, 0.352334579963732, 0.353082520746982, 0.353830613976509, 0.354578947310407, 0.355327596546645, 0.356076627361918, 0.356826096724098, 0.357576054053240, 0.358326542186038, 0.359077598184606, 0.359829254020497]

	if Isobars:

		if Plot_Phi:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')

			plt.plot(T0,Rtildem_P1,'r',ls=ls1,label='Rtildem_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Rtildem_P2,'r',ls=ls2,label='Rtildem_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Rtildem_P3,'r',ls=ls3,label='Rtildem_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,cphis_P1,'m',ls=ls1,label='cphi_s_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,cphis_P2,'m',ls=ls2,label='cphi_s_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,cphis_P3,'m',ls=ls3,label='cphi_s_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,Rtildep0_P1,'b',ls=ls1,label='Rtildep0_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Rtildep0_P2,'b',ls=ls2,label='Rtildep0_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Rtildep0_P3,'b',ls=ls3,label='Rtildep0_{} MPa'.format(P3),lw=linewidth)

			plt.plot(T0,Rtildes0_P1,'k',ls=ls1,label='Rtildes0_{} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,Rtildes0_P2,'k',ls=ls2,label='Rtildes0_{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,Rtildes0_P3,'k',ls=ls3,label='Rtildes0_{} MPa'.format(P3),lw=linewidth)

			# plt.plot(T0,phis0_P1,'y',ls=ls1,label='phi_s0_{} MPa'.format(P1),lw=linewidth)
			# if P2!=0.0:
			# 	plt.plot(T0,phis0_P2,'y',ls=ls2,label='phi_s0_{} MPa'.format(P2),lw=linewidth)
			# if P3!=0.0:
			# 	plt.plot(T0,phis0_P3,'y',ls=ls3,label='phi_s0_{} MPa'.format(P3),lw=linewidth)

			plt.xlabel('Temperature T (K)',fontsize=axis_size)
			plt.ylabel('phi',fontsize=axis_size)
			plt.legend(loc=4,fontsize=size,numpoints=1)
			plt.title(kwargs, fontdict=None, loc='center', pad=None)
			# plt.axis(phi_axes)
			# figS.savefig('./'+output_folder+r'\bin_PS_CO2_Swelling'+img_extension,dpi=img_dpi)

		if Entropy:
			#Plotting the phi's of the PS+CO2 mixture.
			figS = plt.figure(num=None, figsize=(12, 10), dpi=img_dpi, facecolor='w', edgecolor='k')
			# print 'S_1_P1 = ', S_1_P1
			if P1!=0.0:
				plt.plot(T0,S_1_P1,'k',ls=ls2,label='Present Theory at {} MPa'.format(P1),lw=linewidth)
			if P2!=0.0:
				plt.plot(T0,S_1_P2,'k',ls=ls1,label='{} MPa'.format(P2),lw=linewidth)
			if P3!=0.0:
				plt.plot(T0,S_1_P3,'k',ls=ls3,label='{} MPa'.format(P3),lw=linewidth)

			if Plot_S2:
				if P1!=0.0:
					plt.plot(T0,S_2_P1,'m',ls=ls1,label='S_2_{} MPa'.format(P1),lw=linewidth)
				if P2!=0.0:
					plt.plot(T0,S_2_P2,'m',ls=ls2,label='S_2_{} MPa'.format(P2),lw=linewidth)
				if P3!=0.0:
					plt.plot(T0,S_2_P3,'m',ls=ls3,label='S_2_{} MPa'.format(P3),lw=linewidth)
			
			# plt.axvline(x=378,lw=0.5,color='k', linestyle='-.')
			# plt.axhline(y=0.0,lw=0.5,color='k', linestyle='-.')
			# S_max=npy.max(S_1_P1)
			# print 'S_max is:', S_max
			Tg_line=0.0	#xS_infty#0.317*0.98268 #0.310707*0.8708171#0.2361#2.38*0.271##0.2558 #S_max*kx #0.2361
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


if Condo_Original:
	T0 = T0.tolist( )		#To convert array T0 to list T0. In this way Print will give comma between elements
	print 'T0 = ', T0
	print 'Rtildem_P1', Rtildem_P1
	print 'Rtildep0_P1', Rtildep0_P1
	print 'Rtildes0_P1', Rtildes0_P1
	print 'cphip_P1', cphip_P1
	print 'cphis_P1', cphis_P1

	print 'S_1_P1 = ', S_1_P1
	print 'S_1_P2 = ', S_1_P2
	print 'S_1_P3 = ', S_1_P3

if Hassan:
	T0 = T0.tolist( )		#To convert array T0 to list T0. In this way Print will give comma between elements
	print 'T0 = ', T0
	print 'S_1_P1_DHV = ', S_1_P1_DHV
	print 'S_1_P2_DHV = ', S_1_P2_DHV
	print 'S_1_P3_DHV = ', S_1_P3_DHV


if False:	#For sorting refined data
	T0_1 =  [250.0, 250.51724137931035, 251.0344827586207, 251.55172413793105, 252.06896551724137, 252.58620689655172, 253.10344827586206, 253.6206896551724, 254.13793103448276, 254.6551724137931, 255.17241379310346, 255.68965517241378, 256.2068965517241, 256.7241379310345, 257.2413793103448, 257.7586206896552, 258.2758620689655, 258.7931034482759, 259.3103448275862, 259.82758620689657, 260.3448275862069, 260.86206896551727, 261.37931034482756, 261.8965517241379, 262.41379310344826, 262.9310344827586, 263.44827586206895, 263.9655172413793, 264.48275862068965, 265.0, 265.0, 265.1515151515151, 265.3030303030303, 265.45454545454544, 265.6060606060606, 265.75757575757575, 265.90909090909093, 266.06060606060606, 266.2121212121212, 266.3636363636364, 266.5151515151515, 266.6666666666667, 266.8181818181818, 266.969696969697, 267.1212121212121, 267.27272727272725, 267.42424242424244, 267.57575757575756, 267.72727272727275, 267.8787878787879, 268.030303030303, 268.1818181818182, 268.3333333333333, 268.4848484848485, 268.6363636363636, 268.7878787878788, 268.93939393939394, 269.09090909090907, 269.24242424242425, 269.3939393939394, 269.54545454545456, 269.6969696969697, 269.8484848484849, 270.0, 270.1515151515151, 270.3030303030303, 270.45454545454544, 270.6060606060606, 270.75757575757575, 270.90909090909093, 271.06060606060606, 271.2121212121212, 271.3636363636364, 271.5151515151515, 271.6666666666667, 271.8181818181818, 271.969696969697, 272.1212121212121, 272.27272727272725, 272.42424242424244, 272.57575757575756, 272.72727272727275, 272.8787878787879, 273.030303030303, 273.1818181818182, 273.3333333333333, 273.4848484848485, 273.6363636363636, 273.7878787878788, 273.93939393939394, 274.09090909090907, 274.24242424242425, 274.3939393939394, 274.54545454545456, 274.6969696969697, 274.8484848484849, 275.0, 275.1515151515151, 275.3030303030303, 275.45454545454544, 275.6060606060606, 275.75757575757575, 275.90909090909093, 276.06060606060606, 276.2121212121212, 276.3636363636364, 276.5151515151515, 276.6666666666667, 276.8181818181818, 276.969696969697, 277.1212121212121, 277.27272727272725, 277.42424242424244, 277.57575757575756, 277.72727272727275, 277.8787878787879, 278.030303030303, 278.1818181818182, 278.3333333333333, 278.4848484848485, 278.6363636363636, 278.7878787878788, 278.93939393939394, 279.09090909090907, 279.24242424242425, 279.3939393939394, 279.54545454545456, 279.6969696969697, 279.8484848484849, 280.0, 280.0, 280.40816326530614, 280.81632653061223, 281.2244897959184, 281.6326530612245, 282.0408163265306, 282.44897959183675, 282.85714285714283, 283.265306122449, 283.6734693877551, 284.0816326530612, 284.48979591836735, 284.8979591836735, 285.3061224489796, 285.7142857142857, 286.1224489795918, 286.53061224489795, 286.9387755102041, 287.3469387755102, 287.7551020408163, 288.16326530612247, 288.57142857142856, 288.9795918367347, 289.38775510204084, 289.7959183673469, 290.2040816326531, 290.61224489795916, 291.0204081632653, 291.42857142857144, 291.83673469387753, 292.2448979591837, 292.6530612244898, 293.0612244897959, 293.46938775510205, 293.8775510204082, 294.2857142857143, 294.6938775510204, 295.1020408163265, 295.51020408163265, 295.9183673469388, 296.3265306122449, 296.734693877551, 297.14285714285717, 297.55102040816325, 297.9591836734694, 298.36734693877554, 298.7755102040816, 299.18367346938777, 299.59183673469386, 300.0, 300.0, 302.0689655172414, 304.13793103448273, 306.2068965517241, 308.2758620689655, 310.3448275862069, 312.41379310344826, 314.48275862068965, 316.55172413793105, 318.62068965517244, 320.6896551724138, 322.7586206896552, 324.82758620689657, 326.8965517241379, 328.9655172413793, 331.0344827586207, 333.1034482758621, 335.17241379310343, 337.2413793103448, 339.3103448275862, 341.37931034482756, 343.44827586206895, 345.51724137931035, 347.58620689655174, 349.65517241379314, 351.7241379310345, 353.7931034482759, 355.86206896551727, 357.9310344827586, 360.0]
	S_1_P1_DHV_1 =  [0.179067457338083, 0.179896828207601, 0.180727175685738, 0.181558479468713, 0.182390719506330, 0.183223875997577, 0.184057929386331, 0.184892860357178, 0.185728649831339, 0.186565278962702, 0.187402729133951, 0.188240981952788, 0.189080019248264, 0.189919823067179, 0.190760375670587, 0.191601659530381, 0.192443657325959, 0.193286351940967, 0.194129726460129, 0.194973764166140, 0.195818448536642, 0.196663763241262, 0.197509692138641, 0.198356219274027, 0.199203328875682, 0.200051005353016, 0.200899233293536, 0.201747997460348, 0.202597282789640, 0.203447074388210, 0.203447074388210, 0.203696096976233, 0.203945161373979, 0.204194267215251, 0.204443414135065, 0.204692601769638, 0.204941829756394, 0.205191097733950, 0.205440405342113, 0.205689752221881, 0.205939138015429, 0.206188562366112, 0.206438024918455, 0.206687525318154, 0.206937063212063, 0.207186638248198, 0.207436250075727, 0.207685898344966, 0.207935582707377, 0.208185302815562, 0.208435058323255, 0.208684848885325, 0.208934674157765, 0.209184533797690, 0.209434427463333, 0.209684354814040, 0.209934315510264, 0.210184309213566, 0.210434335586605, 0.210684394293133, 0.210934484997999, 0.211184607367136, 0.211434761067561, 0.211684945767369, 0.211935161135730, 0.212185406842887, 0.212435682560147, 0.212685987959879, 0.212936322715513, 0.213186686501531, 0.213437078993466, 0.213687499867899, 0.213937948802451, 0.214188425475782, 0.214438929567588, 0.214689460758595, 0.214940018730554, 0.215190603166242, 0.215441213749452, 0.215691850164995, 0.215942512098691, 0.216193199237370, 0.216443911268864, 0.216694647882007, 0.216945408766628, 0.217196193613549, 0.217447002114584, 0.217697833962528, 0.217948688851162, 0.218199566475242, 0.218450466530502, 0.218701388713645, 0.218952332722342, 0.219203298255228, 0.219454285011900, 0.219705292692911, 0.219956320999767, 0.220207369634926, 0.220458438301792, 0.220709526704712, 0.220960634548973, 0.221211761540800, 0.221462907387349, 0.221714071796708, 0.221965254477891, 0.222216455140836, 0.222467673496399, 0.222718909256354, 0.222970162133387, 0.223221431841103, 0.223472718094000, 0.223724020607489, 0.223975339097880, 0.224226673282380, 0.224478022879091, 0.224729387607006, 0.224980767186007, 0.225232161336859, 0.225483569781210, 0.225734992241589, 0.225986428441397, 0.226237878104912, 0.226489340957278, 0.226740816724509, 0.226992305133481, 0.227243805911931, 0.227495318788455, 0.227746843492503, 0.227998379754378, 0.228249927305230, 0.228249927305230, 0.228927619689436, 0.229605386846871, 0.230283223612624, 0.230961124865395, 0.231639085527101, 0.232317100562486, 0.232995164978739, 0.233673273825110, 0.234351422192544, 0.235029605213301, 0.235707818060605, 0.236386055948269, 0.237064314130355, 0.237742587878083, 0.238420872593127, 0.239099163580004, 0.239777456273001, 0.240455746122208, 0.241134028615919, 0.241812299280300, 0.242490553679075, 0.243168787413201, 0.243846996120559, 0.244525175475641, 0.245203321189244, 0.245881429008170, 0.246559494714920, 0.247237514127382, 0.247915483098648, 0.248593397516500, 0.249271253303347, 0.249949046415835, 0.250626772844586, 0.251304428613921, 0.251982009781590, 0.252659512438499, 0.253336932708445, 0.254014266747849, 0.254691510745495, 0.255368660922274, 0.256045713530925, 0.256722664855783, 0.257399511212528, 0.258076248947940, 0.258752874439649, 0.259429384095895, 0.260105774355291, 0.260782041686577, 0.261458182588395, 0.261458182588395, 0.264883448056960, 0.268304936631702, 0.271722226286569, 0.275134912974034, 0.278542609947439, 0.281944947114926, 0.285341570422798, 0.288732141266310, 0.292116335926102, 0.295493845028660, 0.298864373029291, 0.302227637716288, 0.305583369735024, 0.308931312130860, 0.312271219909812, 0.315602859616036, 0.318926008925237, 0.322240456253206, 0.325546000378721, 0.328842450080125, 0.332129623784937, 0.335407349231888, 0.338675463144839, 0.341933810918041, 0.345182246312274, 0.348420631161380, 0.351648835088786, 0.354866735233613, 0.358074215985969]
	S_1_P2_DHV_1 =  [0.257658334077234, 0.256922942143912, 0.256284411886999, 0.255731351327108, 0.255254366477600, 0.254845607646202, 0.254498439558214, 0.254207196447068, 0.253966996871526, 0.253773601415859, 0.253623301761994, 0.253512833099175, 0.253439304156461, 0.253400140723897, 0.253393039625437, 0.253415930881412, 0.253466946353974, 0.253544393572920, 0.253646733737092, 0.253772563108431, 0.253920597183164, 0.254089657136980, 0.254278658259145, 0.254486599746255, 0.254712556126226, 0.254955669579443, 0.255215143296854, 0.255490235655769, 0.255780255057293, 0.256084555356615, 0.256084555356615, 0.256176309663539, 0.256269222680133, 0.256363280113120, 0.256458467932239, 0.256554772367441, 0.256652179883424, 0.256750677230625, 0.256850251357118, 0.256950889448769, 0.257052578933130, 0.257155307452701, 0.257259062865988, 0.257363833242306, 0.257469606856730, 0.257576372185212, 0.257684117899829, 0.257792832864185, 0.257902506128959, 0.258013126927538, 0.258124684671847, 0.258237168948246, 0.258350569513569, 0.258464876291273, 0.258580079367694, 0.258696168988416, 0.258813135554738, 0.258930969620239, 0.259049661887444, 0.259169203204575, 0.259289584562402, 0.259410797091170, 0.259532832057616, 0.259655680862066, 0.259779335035609, 0.259903786237344, 0.260029026251710, 0.260155046985869, 0.260281840467112, 0.260409398840773, 0.260537714366996, 0.260666779419166, 0.260796586481339, 0.260927128146555, 0.261058397112980, 0.261190386183709, 0.261323088263201, 0.261456496358725, 0.261590603572384, 0.261725403103795, 0.261860888247569, 0.261997052389764, 0.262133889007805, 0.262271391668704, 0.262409554025797, 0.262548369818679, 0.262687832871617, 0.262827937090051, 0.262968676462371, 0.263110045053998, 0.263252037011683, 0.263394646552907, 0.263537867982555, 0.263681695663736, 0.263826124042216, 0.263971147632494, 0.264116761019055, 0.264262958855130, 0.264409735861482, 0.264557086825222, 0.264705006598648, 0.264853490098107, 0.265002532302896, 0.265152128254167, 0.265302273053873, 0.265452961863730, 0.265604189904200, 0.265755952453496, 0.265908244846617, 0.266061062474392, 0.266214400782550, 0.266368255270810, 0.266522621491991, 0.266677495051136, 0.266832871604657, 0.266988746859504, 0.267145116572339, 0.267301976548734, 0.267459322642399, 0.267617150754382, 0.267775456832347, 0.267934236869815, 0.268093486905450, 0.268253203022342, 0.268413381347320, 0.268574018050262, 0.268735109343436, 0.268896651480838, 0.269058640757555, 0.269221073509133, 0.269221073509133, 0.269660825841571, 0.270103700845149, 0.270549631043983, 0.270998550867879, 0.271450396577676, 0.271905106194301, 0.272362619431318, 0.272822877630747, 0.273285823701994, 0.273751402063673, 0.274219558588186, 0.274690240548891, 0.275163396569704, 0.275638976577024, 0.276116931753827, 0.276597214495830, 0.277079778369593, 0.277564578072484, 0.278051569394377, 0.278540709181014, 0.279031955298933, 0.279525266601889, 0.280020602898683, 0.280517924922332, 0.281017194300518, 0.281518373527240, 0.282021425935624, 0.282526315671820, 0.283033007669943, 0.283541467628005, 0.284051661984794, 0.284563557897641, 0.285077123221058, 0.285592326486183, 0.286109136881007, 0.286627524231343, 0.287147458982512, 0.287668912181694, 0.288191855460934, 0.288716261020771, 0.289242101614449, 0.289769350532697, 0.290297981589052, 0.290827969105702, 0.291359287899814, 0.291891913270348, 0.292425820985324, 0.292960987269514, 0.293497388792568, 0.293497388792568, 0.296234575123112, 0.299000110613969, 0.301791417693112, 0.304606136177744, 0.307442098860630, 0.310297310578220, 0.313169930170056, 0.316058254853481, 0.318960706627375, 0.321875820389490, 0.324802233508199, 0.327738676634559, 0.330683965576837, 0.333636994089036, 0.336596727448896, 0.339562196720429, 0.342532493612219, 0.345506765855998, 0.348484213041172, 0.351464082850154, 0.354445667647157, 0.357428301379611, 0.360411356756854, 0.363394242675464, 0.366376401864514, 0.369357308727460, 0.372336467360264, 0.375313409727833, 0.378287693983021]
	S_1_P3_DHV_1 =  [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0.357853665490959, 0.353828643198873, 0.350973390116319, 0.348696917050134, 0.346782232090199, 0.345120815990120, 0.343649206311263, 0.342326512317965, 0.341124574348701, 0.340023032392881, 0.339006607723555, 0.338063490673598, 0.337184333376976, 0.336361588163858, 0.335589058607037, 0.334861584741025, 0.334174816574448, 0.333525047141211, 0.332909088691435, 0.332324174806252, 0.331767888422041, 0.331238102827166, 0.330732935415622, 0.330250710453323, 0.329789928817276, 0.329349243181093, 0.328927437497306, 0.328523409902258, 0.328136158371702, 0.327764768605612, 0.327408403733545, 0.327066295517374, 0.326737736794821, 0.326422074954522, 0.326118706279039, 0.325827071016412, 0.325546649057089, 0.325276956216157, 0.325017540767131, 0.324767980626598, 0.324527880673531, 0.324296870436288, 0.324074602016869, 0.323860748233530, 0.323655000954388, 0.323457069596670, 0.323266679783735, 0.323266679783735, 0.322789341238585, 0.322360301658311, 0.321975564146745, 0.321631630884231, 0.321325421049929, 0.321054205230337, 0.320815552450600, 0.320607286989288, 0.320427452861318, 0.320274284372987, 0.320146181530678, 0.320041689363131, 0.319959480424614, 0.319898339902807, 0.319857152874396, 0.319834892792086, 0.319830614764990, 0.319843441825086, 0.319872563266003, 0.319917225143413, 0.319976727208811, 0.320050415178559, 0.320137678366884, 0.320237945383117, 0.320350680736349, 0.320475381804877, 0.320611576123899, 0.320758818951738, 0.320916691080618, 0.321084796862757, 0.321262762426594, 0.321450234061353, 0.321646876751043, 0.321852372841419, 0.322066420825558, 0.322288734235459, 0.322519040628647, 0.322757080660069, 0.323002607230718, 0.323255384705432, 0.323515188193141, 0.323781802883641, 0.324055023435568, 0.324334653410871, 0.324620504751560, 0.324912397294958, 0.325210158324063, 0.325513622149990, 0.325822629723758, 0.325822629723758, 0.327468925744823, 0.329236271689032, 0.331109928360737, 0.333077525124640, 0.335128564524692, 0.337254051817794, 0.339446213156415, 0.341698277932553, 0.344004308379329, 0.346359064529814, 0.348757896005790, 0.351196654428026, 0.353671621862128, 0.356179451867529, 0.358717120549378, 0.361281885621545, 0.363871251939376, 0.366482942298013, 0.369114872547251, 0.371765130268843, 0.374431956412461, 0.377113729403422, 0.379808951326901, 0.382516235865702, 0.385234297726230, 0.387961943333343, 0.390698062611901, 0.393441621702880, 0.396191656486459]

	T0_2 =  [250.0, 251.42857142857142, 252.85714285714286, 254.28571428571428, 255.71428571428572, 257.14285714285717, 258.57142857142856, 260.0, 261.42857142857144, 262.85714285714283, 264.2857142857143, 265.7142857142857, 267.14285714285717, 268.57142857142856, 270.0, 271.42857142857144, 272.85714285714283, 274.2857142857143, 275.7142857142857, 277.14285714285717, 278.57142857142856, 280.0, 281.42857142857144, 282.8571428571429, 284.2857142857143, 285.7142857142857, 287.14285714285717, 288.57142857142856, 290.0, 291.42857142857144, 292.8571428571429, 294.2857142857143, 295.7142857142857, 297.14285714285717, 298.57142857142856, 300.0, 301.42857142857144, 302.8571428571429, 304.2857142857143, 305.7142857142857, 307.14285714285717, 308.57142857142856, 310.0, 311.42857142857144, 312.8571428571429, 314.2857142857143, 315.7142857142857, 317.1428571428571, 318.57142857142856, 320.0, 320.0, 320.81632653061223, 321.6326530612245, 322.44897959183675, 323.265306122449, 324.0816326530612, 324.8979591836735, 325.7142857142857, 326.53061224489795, 327.3469387755102, 328.16326530612247, 328.9795918367347, 329.7959183673469, 330.61224489795916, 331.42857142857144, 332.2448979591837, 333.0612244897959, 333.8775510204082, 334.6938775510204, 335.51020408163265, 336.3265306122449, 337.14285714285717, 337.9591836734694, 338.7755102040816, 339.59183673469386, 340.40816326530614, 341.2244897959184, 342.0408163265306, 342.85714285714283, 343.6734693877551, 344.48979591836735, 345.3061224489796, 346.1224489795918, 346.9387755102041, 347.7551020408163, 348.57142857142856, 349.38775510204084, 350.2040816326531, 351.0204081632653, 351.83673469387753, 352.6530612244898, 353.46938775510205, 354.2857142857143, 355.1020408163265, 355.9183673469388, 356.734693877551, 357.55102040816325, 358.36734693877554, 359.18367346938777, 360.0]
	S_1_P1_DHV_2 =  [0.179067457338083, 0.181360464329123, 0.183660650477180, 0.185967601865132, 0.188280918464127, 0.190600213501384, 0.192925112870649, 0.195255254581725, 0.197590288245399, 0.199929874592783, 0.202273685020538, 0.204621401169214, 0.206972714522835, 0.209327326033518, 0.211684945767369, 0.214045292570279, 0.216408093752119, 0.218773084787944, 0.221140009034966, 0.223508617464149, 0.225878668405376, 0.228249927305230, 0.230622166496520, 0.232995164978739, 0.235368708208719, 0.237742587878083, 0.240116601835952, 0.242490553679075, 0.244864252804293, 0.247237514127382, 0.249610157945265, 0.251982009781590, 0.254352900238836, 0.256722664855783, 0.259091143970501, 0.261458182588395, 0.263823630255046, 0.266187340933587, 0.268549172886374, 0.270908988560722, 0.273266654478498, 0.275622041129382, 0.277975022867604, 0.280325477811991, 0.282673287749171, 0.285018338039781, 0.287360517527323, 0.289699718451054, 0.292035836358263, 0.294368770023348, 0.294368770023348, 0.295700407442556, 0.297030955178228, 0.298360395704099, 0.299688711811648, 0.301015886605574, 0.302341903499355, 0.303666746210858, 0.304990398758037, 0.306312845454687, 0.307634070906262, 0.308954060005762, 0.310272797929685, 0.311590270134029, 0.312906462350370, 0.314221360581980, 0.315534951100021, 0.316847220439785, 0.318158155396990, 0.319467743024131, 0.320775970626888, 0.322082825760582, 0.323388296226680, 0.324692370069359, 0.325995035572107, 0.327296281254385, 0.328596095868327, 0.329894468395490, 0.331191388043651, 0.332486844243646, 0.333780826646257, 0.335073325119136, 0.336364329743780, 0.337653830812542, 0.338941818825683, 0.340228284488468, 0.341513218708299, 0.342796612591890, 0.344078457442474, 0.345358744757056, 0.346637466223701, 0.347914613718850, 0.349190179304689, 0.350464155226535, 0.351736533910271, 0.353007307959808, 0.354276470154582, 0.355544013447088, 0.356809930960439, 0.358074215985969]
	S_1_P2_DHV_2 =  [0.257658334077234, 0.255855808791684, 0.254656455463316, 0.253907136392810, 0.253508515614807, 0.253392019517903, 0.253508067352293, 0.253819500879150, 0.254297661280222, 0.254919919475590, 0.255668053436319, 0.256527143666158, 0.257484798595636, 0.258530597499046, 0.259655680862066, 0.260852443386098, 0.262114299992286, 0.263435504772646, 0.264811009011975, 0.266236348496590, 0.267707553101932, 0.269221073509133, 0.270773721278562, 0.272362619431317, 0.273985161383223, 0.275638976577024, 0.277321901530699, 0.279031955298933, 0.280767318556213, 0.282526315671820, 0.284307399271899, 0.286109136881007, 0.287930199311909, 0.289769350532697, 0.291625438788374, 0.293497388792568, 0.295384194836048, 0.297284914683972, 0.299198664154280, 0.301124612286544, 0.303061977024441, 0.305010021346535, 0.306968049789610, 0.308935405316770, 0.310911466489255, 0.312895644906541, 0.314887382884064, 0.316886151341973, 0.318891447881730, 0.320902795030357, 0.320902795030357, 0.322054677164543, 0.323208304819342, 0.324363598745772, 0.325520481859477, 0.326678879164688, 0.327838717681705, 0.328999926377672, 0.330162436100489, 0.331326179515658, 0.332491091045929, 0.333657106813568, 0.334824164585128, 0.335992203718576, 0.337161165112649, 0.338330991158342, 0.339501625692387, 0.340673013952656, 0.341845102535351, 0.343017839353934, 0.344191173599672, 0.345365055703741, 0.346539437300799, 0.347714271193974, 0.348889511321174, 0.350065112722681, 0.351241031509952, 0.352417224835577, 0.353593650864339, 0.354770268745328, 0.355947038585055, 0.357123921421526, 0.358300879199239, 0.359477874745040, 0.360654871744832, 0.361831834721075, 0.363008729011056, 0.364185520745887, 0.365362176830210, 0.366538664922575, 0.367714953416453, 0.368891011421877, 0.370066808747673, 0.371242315884251, 0.372417503986955, 0.373592344859922, 0.374766810940453, 0.375940875283867, 0.377114511548810, 0.378287693983021]
	S_1_P3_DHV_2 =  [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0.340800350152404, 0.333081795161268, 0.328467083372334, 0.325391254781664, 0.323266679783735, 0.321798698470641, 0.320815552450600, 0.320207193861723, 0.319898339902807, 0.319834942217659, 0.319976727208811, 0.320292786668601, 0.320758818951738, 0.321355331345533, 0.322066420825558, 0.322878923115147, 0.323781802883641, 0.324765704147108, 0.325822629723758, 0.326945660394133, 0.328128785763402, 0.329366737395422, 0.330654869964424, 0.331989063898841, 0.333365646607911, 0.334781328156563, 0.336233148294912, 0.337718432499200, 0.339234755229560, 0.340779909015835, 0.342351878286566, 0.343948817086057, 0.345569030000035, 0.345569030000035, 0.346504709610835, 0.347447203601973, 0.348396250639769, 0.349351600675059, 0.350313014284768, 0.351280262062069, 0.352253124050838, 0.353231389220552, 0.354214854978164, 0.355203326713842, 0.356196617377747, 0.357194547085320, 0.358196942748761, 0.359203637732616, 0.360214471531592, 0.361229289468855, 0.362247942413261, 0.363270286514089, 0.364296182951968, 0.365325497704810, 0.366358101327661, 0.367393868745474, 0.368432679057888, 0.369474415355170, 0.370518964544560, 0.371566217186294, 0.372616067338675, 0.373668412411561, 0.374723153027744, 0.375780192891680, 0.376839438665130, 0.377900799849233, 0.378964188672638, 0.380029519985309, 0.381096711157640, 0.382165681984585, 0.383236354594470, 0.384308653362235, 0.385382504826823, 0.386457837612495, 0.387534582353823, 0.388612671624170, 0.389692039867448, 0.390772623332973, 0.391854360013251, 0.392937189584524, 0.394021053349939, 0.395105894185179, 0.396191656486459]

	T0 = npy.concatenate([T0_1, T0_2])
	S_1_P1_DHV = npy.concatenate([S_1_P1_DHV_1, S_1_P1_DHV_2])
	S_1_P2_DHV = npy.concatenate([S_1_P2_DHV_1, S_1_P2_DHV_2])
	S_1_P3_DHV = npy.concatenate([S_1_P3_DHV_1, S_1_P3_DHV_2])

	S_1_P1_DHV = npy.array(S_1_P1_DHV)
	S_1_P2_DHV = npy.array(S_1_P2_DHV)
	S_1_P3_DHV = npy.array(S_1_P3_DHV)
	T0 = npy.array(T0)

	inds = T0.argsort()
	sortedS_1_P1_DHV = S_1_P1_DHV[inds]
	sortedS_1_P2_DHV = S_1_P2_DHV[inds]
	sortedS_1_P3_DHV = S_1_P3_DHV[inds]
	sortedT0 = T0[inds]

	T0 = sortedT0.tolist( )		#To convert array T0 to list T0. In this way Print will give comma between elements
	S_1_P1_DHV = sortedS_1_P1_DHV.tolist( )
	S_1_P2_DHV = sortedS_1_P2_DHV.tolist( )
	S_1_P3_DHV = sortedS_1_P3_DHV.tolist( )

	print 'T0 = ', T0
	print 'S_1_P1_DHV = ', S_1_P1_DHV
	print 'S_1_P2_DHV = ', S_1_P2_DHV
	print 'S_1_P3_DHV = ', S_1_P3_DHV