
import os,sys,math,csv,matplotlib.pyplot as plt,numpy as npy
from loadPhysicalConstants import *
from Parameters_of_Different_Polymers import *
from scipy.optimize import *
from inspect import currentframe #To get line number in Print
from To_get_colored_print import *
from math import *
from sympy import *		#This imports "ln"

def density_at_discontinuities(x,r,Ptilde):

	y = Ptilde + (x**2) + (((-2*x)/(((1-(1/r))*(1-x))-1))*(((1-(1/r))*x*(1-x))+((1-x)*ln(1-x))))

	return y

def find_discontinuity_temperature(P,Psstar,Tsstar,Rsstar,Ms):

	# print Psstar,Tsstar,Rsstar
	r = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	# print r

	Ptilde = P/Psstar

	#Let P_max be the maximum pressure above which the solvent exists only in liquid state (only one solution of SL EOS exists) i.e. discontinuity is impossible above that pressure.
	#Then the value of density Rtilde_max at Ptilde_max at such pressure is:
	Rtilde_max = 1/(sqrt(r)+1)
	#And, the value of P_max is:
	Ptilde_max = -1*((Rtilde_max**2) + (((-2*Rtilde_max)/(((1-(1/r))*(1-Rtilde_max))-1))*(((1-(1/r))*Rtilde_max*(1-Rtilde_max))+((1-Rtilde_max)*ln(1-Rtilde_max)))))
	#Above Ptilde only one solution of SL EOS (density) is possible at any temperature. So, P > P_max has no problem and no discontinuity at any temperature.
	#Below Ptilde we can have one or two or three solutions i.e. discontinuity can occure at P < P_max at one temperature.
	#To find the temperature at which discontinuity occurs do following:

	# print Ptilde_max*Psstar

	if Ptilde <= Ptilde_max:
		#Let y = SL EOS, and x = Rtilde
		#At any given pressure: the values of Rtilde at which the slope of y = SL EOS is zero (dy/dx = 0) plus y = 0 (i.e. SL EOS = 0) is:

		R2 = bisect(density_at_discontinuities,0,(1/(sqrt(r)+1)),args=(r,Ptilde))	#This root is not important for us. It is always Rtilde < 1/(sqrt(r)+1)
		# R2 = bisect(density_at_discontinuities,(1/(sqrt(r)+1)),1,args=(r,Ptilde))	#This root is always Rtilde > 1/(sqrt(r)+1)
		# print R1
		# print R2
		# asgash
		# One is negative Rtilde and two are positive Rtilde. Among positive values one value Rtilde < 1/(sqrt(r)+1) and one value Rtilde > 1/(sqrt(r)+1). 
		# Only the value of Rtilde > 1/(sqrt(r)+1) is of our interest.
		R_interest = R2 #Since R2 > 1/(sqrt(r)+1)
		#Then temperatures at which slope dy/dx = 0 of SL EOS and y = 0 (i.e. SL EOS = 0) is:
		Ttilde = (-2*R2)/((1-(1/r))-(1/(1-R2))) #At any given pressure. #This is the temperature at given pressure at which discontinuity will occur.
		#Thus the temperature at which discontinuity will occur is:
		T_discontinuity = Ttilde*Tsstar
		# print T_discontinuity
		# agaasg
		#Above T_discontinuity we have only one phi. i.e. gaseous CO2
		#At and Below T_discontinuity we have three phis. Middle value of phi is incorrect because of maxwell construction.
		#At and Below T_discontinuity we have two remaining phis of liquid and gaseous CO2
		#At and Below T_discontinuity we have take liquid phi as our answer. i.e. phi having the greatest value. 

	elif Ptilde > Ptilde_max:  #Discontinuity is impossible at all temperatures.
		T_discontinuity = float('nan')
		pass

	print 'T_discontinuity', T_discontinuity

	return T_discontinuity


def find_discontinuity_pressure(T,Psstar,Tsstar,Rsstar,Ms):

	# print Psstar,Tsstar,Rsstar
	r = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	# print r

	Ttilde = T/Tsstar

	#Let T_max be the maximum temperature above which the solvent exists only in liquid state (only one solution of SL EOS exists) i.e. discontinuity is impossible above that temperature.
	Ttilde_max = ((2*(1+1/r))-(4/sqrt(r)))/((1-1/r)**2)
	#Above Ttilde_max only one solution of SL EOS (density) is possible at any pressure. So, T > T_max has no problem and no discontinuity at any pressure.
	#Below Ttilde_max we can have one or two or three solutions i.e. discontinuity can occure at T < T_max at one pressure.
	#To find the temperature at which discontinuity occurs do following:

	# print Ttilde_max*Tsstar

	if Ttilde <= Ttilde_max:
		#Let y = SL EOS, and x = Rtilde
		#At any given temperature: the values of Rtilde at which the slope of y = SL EOS is zero (dy/dx = 0) is:

		R2 = ((-((Ttilde*(1-1/r))-2))-(sqrt((((Ttilde*(1-1/r))-2)**2)-(8*Ttilde/r))))/4 #This root is not important for us.
		# R2 = ((-((Ttilde*(1-1/r))-2))+(sqrt((((Ttilde*(1-1/r))-2)**2)-(8*Ttilde/r))))/4 #This root is correct. Becuase we always take greater value of phi as correct one.
		# print R1
		# print R2
		R_interest = R2
		#Then pressure at which slope dy/dx = 0 of SL EOS and y = 0 (i.e. SL EOS = 0) is:
		Ptilde = (((2*R_interest)/((1-(1/r))-(1/(1-R_interest))))*(((1-(1/r))*R_interest)+ln(1-R_interest))) - (R_interest**2) #At any given temperature. #This is the pressure at given temperature at which discontinuity will occur.
		#Thus the temperature at which discontinuity will occur is:
		P_discontinuity = Ptilde*Psstar
		#Below P_discontinuity we have only one phi. i.e. gaseous CO2
		#At and above P_discontinuity we have three phis. Middle value of phi is incorrect because of maxwell construction.
		#At and above P_discontinuity we have two remaining phis of liquid and gaseous CO2
		#At and above P_discontinuity we have take liquid phi as our answer. i.e. phi having the greatest value. 
		#Still the value of P_discontinuity at greater density could be negative that is not possible, thus:
		
		if not math.isnan(P_discontinuity):
			if P_discontinuity < 0:
				P_discontinuity =  float('nan')

	elif Ttilde > Ttilde_max:  #Discontinuity is impossible at all temperatures.
		P_discontinuity = float('nan')
		pass

	print 'P_discontinuity', P_discontinuity

	return P_discontinuity

'''
# For instance:
Polymer_Type='PMMA'
Solvent='CO2'
Parameters_Paper ='Self_Grassia'			# P*T*R* and g,epsilon_2,x (PVT-Tg Data Paper or Direct P*T*R* Values Reference)
Cp_Polymer_Weight = '02kilo_POST_THESIS'	# g,epsilon_2,x (Cp Paper Reference)
Paper_Number = 'Paper15'						# Solubility or Swelling Data Reference
#for PS: 'Paper4_11_12'
#for PMMA: 'Paper15'
kwargs = {'Polymer_Type':Polymer_Type,'Solvent':Solvent,'Parameters_Paper':Parameters_Paper,'Paper_Number':Paper_Number,'Cp_Polymer_Weight':Cp_Polymer_Weight}

Ppstar,Tpstar,Rpstar,Mp,Psstar,Tsstar,Rsstar,Ms,P_exp,Tg_exp=Parameters_of_Different_Polymers(**kwargs)

T = 300
P_answer = find_discontinuity_pressure(T,Psstar,Tsstar,Rsstar,Ms)
print P_answer

P = 3.0
T_answer = find_discontinuity_temperature(P,Psstar,Tsstar,Rsstar,Ms)
print T_answer
'''