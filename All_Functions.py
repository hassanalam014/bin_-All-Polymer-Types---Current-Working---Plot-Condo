import os,math,numpy as npy
from loadPhysicalConstants import *
from checkResults import *
from sympy import *
import warnings
from scipy.optimize import bisect,fsolve
from scipy.interpolate import interp1d
import os,sys,math,matplotlib.pyplot as plt,numpy as npy
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from To_get_colored_print import *
from inspect import currentframe #To get line number in Print
# from p_params import *
# from s_params import *
# import cmath
# import time
# from Parameters_of_Different_Polymers import *

def get_linenumber():
	#To get line number in Print
    cf = currentframe()
    return cf.f_back.f_lineno

def EOS_pure_k(phi_pure_k,P,T,Mk,Pkstar,Tkstar,Rkstar):

	vh_pure_k = kB*Tkstar/Pkstar
	alpha_pure_k = (Pkstar*Mk)/(kB*Tkstar*Rkstar)
	# print alpha_pure_k
	#Equation of state in Pure solvent phase.
	EOS_pure_k = vh_pure_k*P/(kB*T)+(1.0-1.0/alpha_pure_k)*phi_pure_k+ln(1.0-phi_pure_k)+(Tkstar/T)*(phi_pure_k**2)

	return EOS_pure_k

def phi_pure_k_bisect(direction,P,T,Mk,Pkstar,Tkstar,Rkstar):
	
	step_size_of_phi=0.2
	range_max=1/step_size_of_phi

	if direction=='fwd':
		start=0
		end=int(range_max)+1
		step=1
		
	elif direction=='bwd':
		start=int(range_max)
		end=-1
		step=-1

	for i in range(start,end,step):
		k=i*step_size_of_phi

		if k+step*0.000001<0.0:
			guess1=0.000001
		elif k+step*0.000001>1.0:
			guess1=0.999999
		else:
			guess1=k+step*0.000001

		if k+step*0.000001+step*step_size_of_phi<0.0:
			guess2=0.000001
		elif k+step*0.000001+step*step_size_of_phi>1.0:
			guess2=0.999999
		else:
			guess2=k+step*0.000001+step*step_size_of_phi

		phi_pure_k=0.0
		try:
			phi_pure_k = bisect(EOS_pure_k,abs(guess1),abs(guess2),args=(P,T,Mk,Pkstar,Tkstar,Rkstar))
		except:
			# print 'No value found between', k, 'to', k+step_size_of_phi
			pass
		if phi_pure_k!=0.0:
			# print 'Hurry! phi_pure_k is:', phi_pure_k
			break
	
	if phi_pure_k==0.0:
		print 'Program Failed to get value of phi_pure_k in', direction, 'direction'

	return phi_pure_k

def EOS_mix(phi_mix_s,P,T,Mp,Ms,phi_mix_p,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol):
	
	# print 'are you running by any chance'
	vh_pure_p = kB*Tpstar/Ppstar
	vh_pure_s = kB*Tsstar/Psstar

	alpha_pure_p = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_pure_s = (Psstar*Ms)/(kB*Tsstar*Rsstar)

	Tps_star=zeta*math.sqrt(Tpstar*Tsstar)

	if Kier or Hassan:	
		vhm = delta*vh_pure_s									#Hassan: This is definition of delta
		alpha_mix_p = alpha_pure_p*vh_pure_p/vhm				#Hassan: v_h is volume of hole in mixture or pure solvent
		alpha_mix_s = alpha_pure_s*vh_pure_s/vhm							

	elif Condo:
		Rtilde=phi_mix_s+phi_mix_p
		vhm=(phi_mix_s/Rtilde)*vh_pure_s+(phi_mix_p/Rtilde)*vh_pure_p
		alpha_mix_p = alpha_pure_p				#Hassan: v_h is volume of hole in mixture or pure solvent
		alpha_mix_s = alpha_pure_s							

	elif Hassan_Var_Vol:
		Rtilde=phi_mix_s+phi_mix_p
		vhm=(phi_mix_s/Rtilde)*vh_pure_s+(phi_mix_p/Rtilde)*vh_pure_p
		alpha_mix_p = alpha_pure_p*vh_pure_p/vhm				#Hassan: v_h is volume of hole in mixture or pure solvent
		alpha_mix_s = alpha_pure_s*vh_pure_s/vhm	

	#Equation of state in mixture phase.
	EOS_mixture = vhm*P/(kB*T)+(1.0-1.0/alpha_mix_p)*phi_mix_p+(1.0-1.0/alpha_mix_s)*phi_mix_s+ln(1.0-phi_mix_p-phi_mix_s)+(Tsstar/T)*(phi_mix_s**2)+(Tpstar/T)*(phi_mix_p**2)+2*(Tps_star/T)*phi_mix_p*phi_mix_s

	return EOS_mixture

def phi_mix_s_bisect(direction,P,T,Mp,Ms,phi_mix_p,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol):
	
	step_size_of_phi=0.2
	range_max=1/step_size_of_phi

	if direction=='fwd':
		start=0
		end=int(range_max)+1
		step=1
		
	elif direction=='bwd':
		start=int(range_max)
		end=-1
		step=-1

	for i in range(start,end,step):
		k=i*step_size_of_phi

		
		if k+step*0.000001<0.0:
			guess1=0.000001
		elif k+step*0.000001>1.0:
			guess1=0.999999
		else:
			guess1=k+step*0.000001

		if k+step*0.000001+step*step_size_of_phi<0.0:
			guess2=0.000001
		elif k+step*0.000001+step*step_size_of_phi>1.0:
			guess2=0.999999
		else:
			guess2=k+step*0.000001+step*step_size_of_phi


		phi_mix_s=0.0
		try:
			phi_mix_s = bisect(EOS_mix,abs(guess1),abs(guess2),args=(P,T,Mp,Ms,phi_mix_p,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol))
		except:
			# print 'No value found between', k, 'to', k+step_size_of_phi
			pass
		if phi_mix_s!=0.0 and phi_mix_s>0.0 :
			# print 'Hurry! phi_mix_s is:', phi_mix_s
			break
	if phi_mix_s==0.0 or phi_mix_s<0.0:
		phi_mix_s=0.0
		# print 'Program Failed to get value of phi_mix_s at assumed phi_mix_p in', direction, 'direction'
		pass

	return phi_mix_s

def chemicalEquilibrium(phi_mix_p,direction,P,T,phi_pure_s,mu_pure_s,Mp,Ms,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol):

	vh_pure_p = kB*Tpstar/Ppstar
	vh_pure_s = kB*Tsstar/Psstar

	alpha_pure_p = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_pure_s = (Psstar*Ms)/(kB*Tsstar*Rsstar)

	Tps_star=zeta*math.sqrt(Tpstar*Tsstar)

	phi_mix_s=phi_mix_s_bisect(direction,P,T,Mp,Ms,phi_mix_p,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol)

	# ####Falto Addition Starts
	# phi_mix_p_falto=0.81545345
	# phi_mix_s_falto=phi_mix_s_bisect(direction,P,T,Mp,Ms,phi_mix_p_falto,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar)
	# print 'phi_mix_s_falto is:', phi_mix_s_falto, 'at phi_mix_p_falto', phi_mix_p_falto, 'at temperature', T
	# ### Falto Addition Ends

	if Kier or Hassan:	
		vhm = delta*vh_pure_s									#Hassan: This is definition of delta
		alpha_mix_p = alpha_pure_p*vh_pure_p/vhm				#Hassan: v_h is volume of hole in mixture or pure solvent
		alpha_mix_s = alpha_pure_s*vh_pure_s/vhm							

	elif Condo:
		Rtilde=phi_mix_s+phi_mix_p
		vhm=(phi_mix_s/Rtilde)*vh_pure_s+(phi_mix_p/Rtilde)*vh_pure_p
		alpha_mix_p = alpha_pure_p
		alpha_mix_s = alpha_pure_s							

	elif Hassan_Var_Vol:
		Rtilde=phi_mix_s+phi_mix_p
		vhm=(phi_mix_s/Rtilde)*vh_pure_s+(phi_mix_p/Rtilde)*vh_pure_p
		alpha_mix_p = alpha_pure_p*vh_pure_p/vhm				#Hassan: v_h is volume of hole in mixture or pure solvent
		alpha_mix_s = alpha_pure_s*vh_pure_s/vhm	

	#Equilibrium Condition is:
	mu_mix_s=mu_pure_s
	# print 'why are not you runnign'
		
	#Solvent chemical potential in mixture phase.
	if Kier:
		residual_mu= -mu_mix_s+(-1+(1.0/alpha_mix_s)*(0+ln(phi_mix_s))-ln(1-phi_mix_p-phi_mix_s)-2*(Tsstar/T)*phi_mix_s-2*(Tps_star/T)*phi_mix_p) #Kier
	elif Hassan or Hassan_Var_Vol:
		residual_mu= -mu_mix_s+alpha_mix_s*(-1+(1.0/alpha_mix_s)*(1+ln(phi_mix_s))-ln(1-phi_mix_p-phi_mix_s)-2*(Tsstar/T)*phi_mix_s-2*(Tps_star/T)*phi_mix_p) #Including all factors
	elif Condo:
		residual_mu= -mu_mix_s+alpha_mix_s*(-1+(1.0/alpha_mix_s)*(1+ln(phi_mix_s))-ln(1-phi_mix_p-phi_mix_s)-2*(Tsstar/T)*phi_mix_s-2*(Tps_star/T)*phi_mix_p) #Including all factors

	##################Falto Trying Start
	
	# phi_mix_p_falto=0.81545345
	# phi_mix_s_falto=phi_mix_s_bisect(direction,P,T,Mp,Ms,phi_mix_p_falto,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar)
	# Rtilde_falto=phi_mix_p_falto+phi_mix_s_falto
	# residual_mu_falto= -mu_mix_s+alpha_mix_s*(-1+(1.0/alpha_mix_s)*(1+ln(phi_mix_s_falto))-ln(1-phi_mix_p_falto-phi_mix_s_falto)-2*(Tsstar/T)*phi_mix_s_falto-2*(Tps_star/T)*phi_mix_p_falto) #Including all factors
	# print 'residual is', residual_mu_falto, 'at phi_mix_p=', phi_mix_p_falto, 'at phi_mix_s=', phi_mix_s_falto, 'at Rtilde=', Rtilde_falto,'at T', T, 'and P',P
	
	######################Falto Trying Ends

	# print 'residual_mu is', residual_mu
	if phi_mix_s==0.0:
		# print 'Unfortunately program may pick phi_mix_s=', phi_mix_s
		residual_mu=-9000000000000000
	
	# print 'residual is:', residual_mu, 'and mu_pure_s is:', mu_pure_s
	# print 'phi_mix_s for above residual is:', phi_mix_s, 'and phi_mix_p:', phi_mix_p, 'and phi_pure_s:', phi_pure_s
	return residual_mu

def chemicalEquilibrium_bisect(direction,P,T,phi_pure_s,mu_pure_s,Mp,Ms,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol):
	
	step_size_refinement=[0.2,0.1,0.01]

	for m in range(0,len(step_size_refinement)):	

		step_size_of_phi=step_size_refinement[m]
		range_max=1/step_size_of_phi

		if direction=='fwd':
			start=0
			end=int(range_max)+1
			step=1
			# print 'forward'
			
		elif direction=='bwd':
			start=int(range_max)
			end=-1
			step=-1
			# print 'backward'

		for i in range(start,end,step):
			k=i*step_size_of_phi
					
			if k+step*0.000001<0.0:
				guess1=0.000001
			elif k+step*0.000001>1.0:
				guess1=0.999999
			else:
				guess1=k+step*0.000001

			if k+step*0.000001+step*step_size_of_phi<0.0:
				guess2=0.000001
			elif k+step*0.000001+step*step_size_of_phi>1.0:
				guess2=0.999999
			else:
				guess2=k+step*0.000001+step*step_size_of_phi
			# print guess1
			# print guess2
			# print '------'
			# print 'P=',P,'T=',T,'phi_pure_s=',phi_pure_s,'mu_pure_s=',mu_pure_s
			phi_mix_p=0.0
			try:
				phi_mix_p = bisect(chemicalEquilibrium,abs(guess1),abs(guess2),args=(direction,P,T,phi_pure_s,mu_pure_s,Mp,Ms,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol))
			except:
				# print 'No value found between', k+step*0.000001, 'to', k+step*0.000001+step*step_size_of_phi
				pass
			if phi_mix_p!=0.0 and phi_mix_p>0.0:
				# print 'Hurry! phi_mix_p is:', phi_mix_p
				break
		if phi_mix_p!=0.0 and phi_mix_p>0.0:
			# print 'Hurry! phi_mix_p is:', phi_mix_p
			break
		elif phi_mix_p==0.0 or phi_mix_p<0.0:
			# print 'Program Failed to get value of phi_mix_p in', direction, 'direction'
			pass
	if phi_mix_p==0.0 or phi_mix_p<0.0:
		# print 'this is failing'
		print 'Program Failed to get value of phi_mix_p in', direction, 'direction'

	phi_mix_s=phi_mix_s_bisect(direction,P,T,Mp,Ms,phi_mix_p,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol)
	
	#########################################################
	vh_pure_p = kB*Tpstar/Ppstar
	vh_pure_s = kB*Tsstar/Psstar
	alpha_pure_p = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_pure_s = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	Tps_star=zeta*math.sqrt(Tpstar*Tsstar)
	if Kier or Hassan:	
		vhm = delta*vh_pure_s									#Hassan: This is definition of delta
		alpha_mix_p = alpha_pure_p*vh_pure_p/vhm				#Hassan: v_h is volume of hole in mixture or pure solvent
		alpha_mix_s = alpha_pure_s*vh_pure_s/vhm
	elif Condo:
		Rtilde=phi_mix_s+phi_mix_p
		vhm=(phi_mix_s/Rtilde)*vh_pure_s+(phi_mix_p/Rtilde)*vh_pure_p
	elif Hassan_Var_Vol:
		Rtilde=phi_mix_s+phi_mix_p
		vhm=(phi_mix_s/Rtilde)*vh_pure_s+(phi_mix_p/Rtilde)*vh_pure_p
		alpha_mix_p = alpha_pure_p*vh_pure_p/vhm				#Hassan: v_h is volume of hole in mixture or pure solvent
		alpha_mix_s = alpha_pure_s*vh_pure_s/vhm	

	mu_mix_s=mu_pure_s
	if Kier:
		residual_mu= -mu_mix_s+(-1+(1.0/alpha_mix_s)*(0+ln(phi_mix_s))-ln(1-phi_mix_p-phi_mix_s)-2*(Tsstar/T)*phi_mix_s-2*(Tps_star/T)*phi_mix_p) #Kier
	elif Hassan or Hassan_Var_Vol:
		residual_mu= -mu_mix_s+alpha_mix_s*(-1+(1.0/alpha_mix_s)*(1+ln(phi_mix_s))-ln(1-phi_mix_p-phi_mix_s)-2*(Tsstar/T)*phi_mix_s-2*(Tps_star/T)*phi_mix_p) #Including all factors
	elif Condo:
		residual_mu= -mu_mix_s+alpha_mix_s*(-1+(1.0/alpha_mix_s)*(1+ln(phi_mix_s))-ln(1-phi_mix_p-phi_mix_s)-2*(Tsstar/T)*phi_mix_s-2*(Tps_star/T)*phi_mix_p) #Including all factors

	print '#####################################################'
	print 'residual is:', residual_mu
	# print 'phi_mix_s for above residual is:', phi_mix_s
	print '#####################################################'
	##########################################################

	if phi_mix_s==0.0:
		print 'Program Failed to get value of phi_mix_s in', direction, 'direction'

	return phi_mix_p,phi_mix_s

def binaryPhaseEquilibriumCHV_bisect(direction,P,T,Mp,Ms,**kwargs):

	#Reference:
	# -p --> polymer
	# -s --> solvent

	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))
	
	#PURE FLUID PARAMETERS.
	alpha_pure_s = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	
	phi_pure_s=phi_pure_k_bisect(direction,P,T,Ms,Psstar,Tsstar,Rsstar)
	phi_pure_p=phi_pure_k_bisect(direction,P,T,Mp,Ppstar,Tpstar,Rpstar)

	# print phi_pure_s
	# print phi_pure_p
	# print Kier
	# print Hassan 
	# print Condo
	# print delta
	# print zeta

	#CHECKING IF PURE VOLUME FRACTION RESULTS ARE VALID.
	checkVolumeFraction(phi_pure_s,'phi_s')
	checkVolumeFraction(phi_pure_p,'phi_p')

	# print phi_pure_s
	#Solvent Chemical Potential in pure solvent phase.
	if Kier:
		mu_pure_s = (-1+(1.0/alpha_pure_s)*(0+ln(phi_pure_s))-ln(1-phi_pure_s)-2*(Tsstar/T)*phi_pure_s) #Kier
		# print mu_pure_s
	elif Hassan or Hassan_Var_Vol:
		mu_pure_s = alpha_pure_s*(-1+(1.0/alpha_pure_s)*(1+ln(phi_pure_s))-ln(1-phi_pure_s)-2*(Tsstar/T)*phi_pure_s) #Including 1 and prefactor
	elif Condo:			#Same as Hassan
		mu_pure_s = alpha_pure_s*(-1+(1.0/alpha_pure_s)*(1+ln(phi_pure_s))-ln(1-phi_pure_s)-2*(Tsstar/T)*phi_pure_s) #Including 1 and prefactor
		# print 'Kier mu_pure_s is:', mu_pure_s, 'at temperature=', T, 'and phi_pure_s:', phi_pure_s
	phi_mix_p,phi_mix_s=chemicalEquilibrium_bisect(direction,P,T,phi_pure_s,mu_pure_s,Mp,Ms,delta,zeta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar,Kier,Hassan,Condo,Hassan_Var_Vol)

	#CHECKING IF MIXTURE VOLUME FRACTION RESULTS ARE VALID.
	checkVolumeFraction([phi_mix_p,phi_mix_s],['phi_p','phi_s'])

	return [P,T,phi_mix_p,phi_mix_s,phi_pure_p,phi_pure_s]

#Bisect Method Original Condo

def EOS_Original(Rtilde,Ptilde,Ttilde,r):

	#Equation of state in Pure solvent phase.
	EOS=Rtilde**2+Ptilde+Ttilde*(ln(1-Rtilde)+(1-1/r)*Rtilde)

	return EOS

def EOS_Original_bisect(direction,Ptilde,Ttilde,r):
	
	step_size=0.2
	range_max=1/step_size

	if direction=='fwd':
		start=0
		end=int(range_max)+1
		step=1
		
	elif direction=='bwd':
		start=int(range_max)
		end=-1
		step=-1

	for i in range(start,end,step):
		k=i*step_size

		
		if k+step*0.001<0.0:
			guess1=0.001
		elif k+step*0.001>1.0:
			guess1=0.999
		else:
			guess1=k+step*0.001

		if k+step*0.001+step*step_size<0.0:
			guess2=0.001
		elif k+step*0.001+step*step_size>1.0:
			guess2=0.999
		else:
			guess2=k+step*0.001+step*step_size

		Rtilde=0.0
		try:
			Rtilde = bisect(EOS_Original,abs(guess1),abs(guess2),args=(Ptilde,Ttilde,r))
		except:
			# print 'No value found between', k, 'to', k+step_size
			pass
		if Rtilde!=0.0:
			# print 'Hurry! Rtilde is:', Rtilde
			break
	
	if Rtilde==0.0:
		print 'Program Failed to get value of Rtilde in', direction, 'direction'

	return Rtilde

def chemicalEquilibrium_Original(phi_s,direction,mus0,P,T,Psstar,Tsstar,Rsstar,Ppstar,Tpstar,Rpstar,rs,rp,Xsp):

	phi_p=1-phi_s
	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar
	vhm=phi_s*vhs+phi_p*vhp
	Tstar=phi_s*Tsstar+phi_p*Tpstar-phi_s*phi_p*T*Xsp
	Pstar=kB*Tstar/vhm

	Ptilde=P/Pstar
	Ttilde=T/Tstar
	r=1/(phi_s/rs+phi_p/rp)
	Rtilde=EOS_Original_bisect(direction,Ptilde,Ttilde,r)
	vtilde=1/Rtilde
	Pstilde=P/Psstar
	Tstilde=T/Tsstar

	#Equilibrium Condition is:
	musm=mus0

	#Solvent chemical potential in mixture phase.
	residual_mu= -musm+(ln(phi_s)+(1-rs/rp)*phi_p+rs*Rtilde*Xsp*phi_p**2+rs*((-1*Rtilde+Pstilde*vtilde)/Tstilde+(vtilde-1)*ln(1-Rtilde)+ln(Rtilde)/rs))

	# ##################Falto Trying Start
	# phi_s_falto=phi_s
	# phi_p_falto=1-phi_s_falto
	# vhm_falto=phi_s_falto*vhs+phi_p_falto*vhp
	# Tstar_falto=phi_s_falto*Tsstar+phi_p_falto*Tpstar-phi_s_falto*phi_p_falto*T*Xsp
	# Pstar_falto=kB*Tstar_falto/vhm_falto

	# Ptilde_falto=P/Pstar_falto
	# Ttilde_falto=T/Tstar_falto
	# r_falto=1/(phi_s_falto/rs+phi_p_falto/rp)
	# Rtilde_falto=EOS_Original_bisect(direction,Ptilde_falto,Ttilde_falto,r_falto)
	# vtilde_falto=1/Rtilde_falto
	# Pstilde=P/Psstar
	# Tstilde=T/Tsstar

	# #Equilibrium Condition is:
	# # musm=mus0
	# musm=0
	# # kphi_p_falto=0.81545345
	# # Rtilde_falto=0.909237527009
	# # phi_p_falto=kphi_p_falto/Rtilde_falto
	# # phi_s_falto=1.0-phi_p_falto
	# # # r_falto=1/(phi_s_falto/rs+phi_p_falto/rp)
	# # # Rtilde_falto=EOS_Original_bisect(direction,Ptilde,Ttilde,r_falto)
	# # vtilde_falto=1/Rtilde_falto
	# cesidual_mu_falto= -musm+(ln(phi_s_falto)+(1-rs/rp)*phi_p_falto+rs*Rtilde_falto*Xsp*phi_p_falto**2+rs*((-1*Rtilde_falto+Pstilde*vtilde_falto)/Tstilde+(vtilde_falto-1)*ln(1-Rtilde_falto)+ln(Rtilde_falto)/rs))
	# kphi_p_falto=phi_p_falto*Rtilde_falto
	# kphi_s_falto=phi_s_falto*Rtilde_falto
	# Tps_star=zeta*math.sqrt(Tpstar*Tsstar)
	# alpha_0=vhm_falto/vhs
	# print alpha_0
	# residual_mu_falto= -musm+rs*(-1/alpha_0+(1.0/rs)*(1+ln(kphi_s_falto))-(1/alpha_0)*ln(1-kphi_p_falto-kphi_s_falto)-2*(Tsstar/T)*kphi_s_falto-2*(Tps_star/T)*kphi_p_falto) #Including all factors
	# # residual_mu_falto= -musm+rs*(-1+(1.0/rs)*(0+ln(kphi_s_falto))-ln(1-kphi_p_falto-kphi_s_falto)-2*(Tsstar/T)*kphi_s_falto-2*(Tps_star/T)*kphi_p_falto) #Kier

	# print 'cesidual is', cesidual_mu_falto, 'at phi_p_falto=', kphi_p_falto, 'at phi_s_falto=', kphi_s_falto, 'at Rtilde=', Rtilde_falto,'at T', T, 'and P',P
	# print 'residual is', residual_mu_falto, 'at phi_p_falto=', kphi_p_falto, 'at phi_s_falto=', kphi_s_falto, 'at Rtilde=', Rtilde_falto,'at T', T, 'and P',P

	######################Falto Trying Ends



	if Rtilde==0.0:
		# print 'Unfortunately Rtilde can be be calulated at this guess phi_s:', phi_s
		residual_mu=-9000000000000000
	
	# print 'residual_mu is:', residual_mu
	# print 'phi_s for above residual is:', phi_s
	return residual_mu

def chemicalEquilibrium_Original_bisect(direction,mus0,P,T,Psstar,Tsstar,Rsstar,Ppstar,Tpstar,Rpstar,rs,rp,Xsp):
	
	step_size=0.2
	range_max=1/step_size

	if direction=='fwd':
		start=0
		end=int(range_max)+1
		step=1
		# print 'forward'
		
	elif direction=='bwd':
		start=int(range_max)
		end=-1
		step=-1
		# print 'backward'

	for i in range(start,end,step):
		k=i*step_size
				
		if k+step*0.001<0.0:
			guess1=0.001
		elif k+step*0.001>1.0:
			guess1=0.999
		else:
			guess1=k+step*0.001

		if k+step*0.001+step*step_size<0.0:
			guess2=0.001
		elif k+step*0.001+step*step_size>1.0:
			guess2=0.999
		else:
			guess2=k+step*0.001+step*step_size

		phi_s=0.0
		try:
			phi_s = bisect(chemicalEquilibrium_Original,abs(guess1),abs(guess2),args=(direction,mus0,P,T,Psstar,Tsstar,Rsstar,Ppstar,Tpstar,Rpstar,rs,rp,Xsp))
		except:
			# print 'No value found between', k, 'to', k+step_size
			pass
		if phi_s!=0.0 and phi_s>0.0:
			# print 'Hurry! phi_s is:', phi_s
			break

	if phi_s==0.0 or phi_s<0.0:
		print 'Program Failed to get value of phi_s in', direction, 'direction'
	# print phi_s
	return phi_s

def binaryPhaseEquilibriumCondo_Original_bisect(direction,P,T,Mp,Ms,**kwargs):

	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))
	
	#PURE FLUID PARAMETERS.
	rs = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	rp = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	
	Pstilde=P/Psstar
	Tstilde=T/Tsstar

	# print Psstar
	# print Tsstar
	# print Rsstar
	# print rs
	# print Ms

	Rstilde=EOS_Original_bisect(direction,Pstilde,Tstilde,rs)

	Pptilde=P/Ppstar
	Tptilde=T/Tpstar
	Rptilde=EOS_Original_bisect(direction,Pptilde,Tptilde,rp)
	# print Rstilde
	vstilde=1/Rstilde

	#Solvent chemical potential in mixture phase.
	mus0=rs*((-1*Rstilde+Pstilde*vstilde)/Tstilde+(vstilde-1)*ln(1-Rstilde)+ln(Rstilde)/rs)
	# print 'condo mu_pure_s is:', mus0, 'at temperature=', T, 'and phi_pure_s:', Rstilde

	Tspstar=zeta*math.sqrt(Tsstar*Tpstar)
	Xsp=(Tsstar+Tpstar-2*Tspstar)/T

	# R=Rstilde*Rsstar
	phi_s=chemicalEquilibrium_Original_bisect(direction,mus0,P,T,Psstar,Tsstar,Rsstar,Ppstar,Tpstar,Rpstar,rs,rp,Xsp)

	phi_p=1-phi_s
	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar
	vhm=phi_s*vhs+phi_p*vhp
	Tstar=phi_s*Tsstar+phi_p*Tpstar-phi_s*phi_p*T*Xsp
	Pstar=kB*Tstar/vhm

	Ptilde=P/Pstar
	Ttilde=T/Tstar
	r=1/(phi_s/rs+phi_p/rp)
	Rtilde=EOS_Original_bisect(direction,Ptilde,Ttilde,r)

	return [phi_s,phi_p,Rtilde,Rstilde,Rptilde]

###########################################################################
###########################################################################
#Entropy Functions

def ThermodynamicVariables(P,T,phi_p,phi_s,Rptilde,Mp,Ms,**kwargs):

	for key,value in kwargs.items():
		exec "%s=%s" % (key,value)
	
	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar
	alpha_p0 = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_s0 = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	# vhm = delta*vhs
	Vratio=1.0

	##########################################################################
	# Vratio=vhp/vhm
	# F_not valid=(g*exp(-Vratio*epsilon_p/(kB*T)))/(1+g*exp(-Vratio*epsilon_p/(kB*T)))
	# S_2_wrong=(Ppstar/(Rpstar*Tpstar))*(-((Vratio/phi_p)*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((Vratio/alpha_s)*(phi_s/phi_p)*(ln(phi_s)))-((Vratio/alpha_p)*(ln(phi_p)))+((F*Vratio*epsilon_p)/(kB*T))-(ln(1-F)))
	# S_1_wrong=(Ppstar/(Rpstar*Tpstar))*(-((Vratio/phi_p)*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((Vratio/alpha_s)*(phi_s/phi_p)*(ln(phi_s)))-((Vratio/alpha_p)*(ln(phi_p)))+((F*Vratio*epsilon_p)/(kB*T))-(ln(1-F)))
	# S_term1_wrong=(Ppstar/(Rpstar*Tpstar))*(-((Vratio/phi_p)*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p))))
	# S_term2_wrong=(Ppstar/(Rpstar*Tpstar))*(-((Vratio/alpha_s)*(phi_s/phi_p)*(ln(phi_s))))
	# S_term3_wrong=(Ppstar/(Rpstar*Tpstar))*(-((Vratio/alpha_p)*(ln(phi_p))))
	# S_term4_wrong=(Ppstar/(Rpstar*Tpstar))*(((F*Vratio*epsilon_p)/(kB*T)))
	# S_term5_wrong=(Ppstar/(Rpstar*Tpstar))*(-(ln(1-F)))
	# S_2_pure_wrong=(Ppstar/(Rpstar*Tpstar))*(-((1-Rptilde)*(ln(1-Rptilde))/Rptilde)-((ln(Rptilde))/alpha_p0)+((Tpstar/T)*g*(exp(-((Vratio*epsilon_p))/(kB*T)))/(1+g*exp(-((Vratio*epsilon_p))/(kB*T))))+((1/Vratio)*ln(1+g*exp(-(Vratio*epsilon_p)/(kB*T)))))
	###########################################################################
	
	# if P==0.101325:
	# 	print phi_s,phi_p,T,P

	#Following are My Theory Equations with, in general, v!=v_0:
	F=(g*exp(-Vratio*epsilon_p/(kB*T)))/(1+g*exp(-Vratio*epsilon_p/(kB*T)))
	# S_1=(Ppstar/(Rpstar*Tpstar))*(-((vhp/(vhm*phi_p))*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((1/(alpha_s))*(phi_s/phi_p)*(ln(phi_s)))-((1/(alpha_p))*(ln(phi_p)))+((F*epsilon_p)/(kB*T))-((1/Vratio)*ln(1-F)))
	# S_1=(Ppstar/(Rpstar*Tpstar))*(-((1/(delta_p*phi_p))*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((1/(delta_p*alpha_s))*(phi_s/phi_p)*(ln(phi_s)))-((1/(delta_p*alpha_p))*(ln(phi_p)))+((F*epsilon_p)/(delta_p*kB*T))-(ln(1-F)))
	# S_1=(Ppstar/(Rpstar*Tpstar))*(-((vhp/(vhm*phi_p))*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((vhp/(alpha_s*vhm))*(phi_s/phi_p)*(ln(phi_s)))-((vhp/(vhm*alpha_p))*(ln(phi_p)))+((F*Vratio*epsilon_p)/(kB*T))-((1/Vratio)*ln(1-F)))
	
	# In below equation v_r is taken as vhp:
	# Seems wrong: S_1=(Ppstar/(Rpstar*Tpstar))*(-((1/(phi_p))*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((vhp/(alpha_s0))*(phi_s/phi_p)*(ln(phi_s)))-((1/(alpha_p0))*(ln(phi_p)))+((F*epsilon_p)/(kB*T))-((1/Vratio)*ln(1-F)))

	#Below assuming system to be in pure vhp state but phi_s present some how:
	alpha_s = alpha_s0*vhs/vhp
	S_1=(Ppstar/(Rpstar*Tpstar))*(-((1/(phi_p))*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((1/(alpha_s))*(phi_s/phi_p)*(ln(phi_s)))-((1/(alpha_p0))*(ln(phi_p)))+((F*epsilon_p)/(kB*T))-((1/Vratio)*ln(1-F)))
	S_2_slightly_approx_by_alpha_s0=(Ppstar/(Rpstar*Tpstar))*(-((1/(phi_p))*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((1/(alpha_s0))*(phi_s/phi_p)*(ln(phi_s)))-((1/(alpha_p0))*(ln(phi_p)))+((F*epsilon_p)/(kB*T))-((1/Vratio)*ln(1-F)))
	S_2=S_2_slightly_approx_by_alpha_s0
	# For Pure:
	Vratio=1.0
	S_2_pure=(Ppstar/(Rpstar*Tpstar))*(-((1-Rptilde)*(ln(1-Rptilde))/Rptilde)-((ln(Rptilde))/alpha_p0)+((epsilon_p/(kB*T))*g*(exp(-((Vratio*epsilon_p))/(kB*T)))/(1+g*exp(-((Vratio*epsilon_p))/(kB*T))))+((1/Vratio)*ln(1+g*exp(-(Vratio*epsilon_p)/(kB*T)))))

	# S_2=S_2_pure

	# Old Entropy:
	vhm = delta*vhs
	alpha_p = alpha_p0*vhp/vhm
	alpha_s = alpha_s0*vhs/vhm
	Vratio=1.0
	F=(g*exp(-Vratio*epsilon_p/(kB*T)))/(1+g*exp(-Vratio*epsilon_p/(kB*T)))
	S_worked=(Ppstar/(Rpstar*Tpstar))*(-((Vratio/phi_p)*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((Vratio/alpha_s)*(phi_s/phi_p)*(ln(phi_s)))-((Vratio/alpha_p)*(ln(phi_p)))+((F*Vratio*epsilon_p)/(kB*T))-(ln(1-F)))
	# S_2=S_worked

	#New: For "Hassan_Var_Vol" i.e. variable hole volume
	F=(g*exp(-Vratio*epsilon_p/(kB*T)))/(1+g*exp(-Vratio*epsilon_p/(kB*T)))	
	Rtilde=phi_s+phi_p
	vhm=(phi_s/Rtilde)*vhs+(phi_p/Rtilde)*vhp
	alpha_p = alpha_p0*vhp/vhm				#Hassan: v_h is volume of hole in mixture or pure solvent
	alpha_s = alpha_s0*vhs/vhm
	S_2=(Ppstar/(Rpstar*Tpstar))*(-((1/(phi_p))*(1-phi_s-phi_p)*(ln(1-phi_s-phi_p)))-((1/(alpha_s))*(phi_s/phi_p)*(ln(phi_s)))-((1/(alpha_p))*(ln(phi_p)))+((F*epsilon_p)/(kB*T))-((1/Vratio)*ln(1-F)))

	# prRed('S_1={},P={},T={},phip={},phis={},phi_p0={},Mp={},Ms={},g={},epsilon_p={},zeta={},delta={},Ppstar={},Tpstar={},Rpstar={},Psstar={},Tsstar={},Rsstar={}'.format(S_1,P,T,phi_p,phi_s,Rptilde,Mp,Ms,g,epsilon_p,zeta,delta,Ppstar,Tpstar,Rpstar,Psstar,Tsstar,Rsstar))

	# print 'S_1=',S_1,'P=',P,'T=',T,'phip=',phi_p,'phis=',phi_s,'phi_p0=',Rptilde,'Mp=',Mp,'Ms=',Ms,'g=',g,'epsilon_p=',epsilon_p,'zeta=',zeta,'delta=',delta,'Ppstar=',Ppstar,'Tpstar=',Tpstar,'Rpstar=',Rpstar,'Psstar=',Psstar,'Tsstar=',Tsstar,'Rsstar=',Rsstar
	if phi_p == 'nan' or phi_s == 'nan':
		S_1 = float('nan')
		S_2 = float('nan')

	return [S_1,S_2]

def CondoThermodynamicVariables_Modified(P,T,phi_p,phi_s,Mp,Ms,**kwargs):
	
	for key,value in kwargs.items():
		exec "%s=%s" % (key,value)
	
	alpha_p0=(Mp*Ppstar)/(kB*Rpstar*Tpstar)
	alpha_s0=(Ms*Psstar)/(kB*Rsstar*Tsstar)

	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar

	# Tps_star=zeta*math.sqrt(Tpstar*Tsstar)

	Rtilde=phi_s+phi_p
	vhm=(phi_s/Rtilde)*vhs+(phi_p/Rtilde)*vhp
	alpha_pm = alpha_p0
	alpha_sm = alpha_s0					

	r=Rtilde/((phi_p/alpha_pm)+(phi_s/alpha_sm))					

	Fp=(((z-2)*exp(-epsilon_p/(kB*T)))/(1+(z-2)*exp(-epsilon_p/(kB*T))))
	Fs=(((z-2)*exp(-epsilon_s/(kB*T)))/(1+(z-2)*exp(-epsilon_s/(kB*T))))

	S_1=(Ppstar/(Rpstar*Tpstar))*(-((1-Rtilde)*ln(1-Rtilde)/phi_p)-(ln(phi_p)/alpha_pm)+(ln(alpha_pm)/alpha_pm)-(phi_s*ln(phi_s)/phi_p)-(1+((ln(2/z)-1)/r))-(((alpha_pm-2)/alpha_pm)*(ln(1-Fp)-(Fp*epsilon_p/(kB*T))))-((phi_s/phi_p)*(-ln(1-Fs)+(Fs*epsilon_s/(kB*T))+1+((ln(2/z)-1)/r))))

	# -((1-Rtilde)*ln(1-Rtilde)/phi_p)-(ln(phi_p)/alpha_pm)+(ln(alpha_pm)/alpha_pm)-(phi_s*ln(phi_s)/phi_p)-(1+((ln(2/z)-1)/r))-(((alpha_pm-2)/alpha_pm)*(ln(1-Fp)-(Fp*epsilon_p/(kB*T))))-((phi_s/phi_p)*(-ln(1-Fs)+(Fs*epsilon_s/(kB*T))+1+((ln(2/z)-1)/r)))

	S_term1=(Ppstar/(Rpstar*Tpstar))*(-((1-Rtilde)*ln(1-Rtilde)/phi_p))
	S_term2=(Ppstar/(Rpstar*Tpstar))*(-(phi_s*ln(phi_s)/phi_p))
	S_term3=(Ppstar/(Rpstar*Tpstar))*(-(ln(phi_p)/alpha_pm))
	S_term4=(Ppstar/(Rpstar*Tpstar))*(-(((alpha_pm-2)/alpha_pm)*(-(Fp*epsilon_p/(kB*T)))))
	S_term5=(Ppstar/(Rpstar*Tpstar))*(-(((alpha_pm-2)/alpha_pm)*(ln(1-Fp))))
	S_term6=(Ppstar/(Rpstar*Tpstar))*(-((phi_s/phi_p)*(-ln(1-Fs)+(Fs*epsilon_s/(kB*T))+1+((ln(2/z)-1)/r))))
	S_term7=(Ppstar/(Rpstar*Tpstar))*(-(((ln(2/z)-1)/r)))

	vtilde=1/Rtilde
	S_2=(Ppstar/(Rpstar*Tpstar))*(1/(phi_p/Rtilde))*(-1*((vtilde-1)*ln(1-Rtilde)+ln(Rtilde)/r+((phi_s/Rtilde)/alpha_sm)*ln((phi_s/Rtilde)/alpha_sm)+((phi_p/Rtilde)/alpha_pm)*ln((phi_p/Rtilde)/alpha_pm)+1+(ln(2/z)-1)/r+((phi_s/Rtilde)/alpha_sm)*(alpha_sm-2)*(ln(1-Fs)-(Fs*epsilon_s/(kB*T)))+((phi_p/Rtilde)/alpha_pm)*(alpha_pm-2)*(ln(1-Fp)-(Fp*epsilon_p/(kB*T)))))

	return S_1,S_2

def CondoThermodynamicVariables_Original(P,T,Rtilde,phi_s,Mp,Ms,**kwargs):
			
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

	vtilde=1/Rtilde

	# Pstilde=P/Psstar
	# Tstilde=T/Tsstar	
	# Pptilde=P/Ppstar
	# Tptilde=T/Tpstar

	Fp=((z-2)*exp(-epsilon_p/(kB*T)))/(1+(z-2)*exp(-epsilon_p/(kB*T)))
	Fs=((z-2)*exp(-epsilon_s/(kB*T)))/(1+(z-2)*exp(-epsilon_s/(kB*T)))

	S_1=-1*((vtilde-1)*ln(1-Rtilde)+ln(Rtilde)/r+(phi_s/rs)*ln(phi_s/rs)+(phi_p/rp)*ln(phi_p/rp)+1+(ln(2/z)-1)/r+(phi_s/rs)*(rs-2)*(ln(1-Fs)-(Fs*epsilon_s/(kB*T)))+(phi_p/rp)*(rp-2)*(ln(1-Fp)-(Fp*epsilon_p/(kB*T))))
	S_2=(Ppstar/(Rpstar*Tpstar))*(1/phi_p)*(-1*((vtilde-1)*ln(1-Rtilde)+ln(Rtilde)/r+(phi_s/rs)*ln(phi_s/rs)+(phi_p/rp)*ln(phi_p/rp)+1+(ln(2/z)-1)/r+(phi_s/rs)*(rs-2)*(ln(1-Fs)-(Fs*epsilon_s/(kB*T)))+(phi_p/rp)*(rp-2)*(ln(1-Fp)-(Fp*epsilon_p/(kB*T)))))

	return [S_1,S_2]

###########################################################################

def binarySolubilitySwellingCHV(P,T,Mp,Ms,**kwargs):

	# Xs:the total solubility m of solvent and co-solvent.
	# Sw:the volume swelling.

	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))

	#PURE FLUID PARAMETERS.
	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar
	alpha_p0 = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_s0 = (Psstar*Ms)/(kB*Tsstar*Rsstar)

	# CALCULATION OF VOLUME FRACTIONS AT P, T.

	if forward:
		[Pd,Td,phip,phis,phip0,phis0] = binaryPhaseEquilibriumCHV_bisect('fwd',P,T,Mp,Ms,**kwargs)
	if backward:
		[Pd,Td,phip,phis,phip0,phis0] = binaryPhaseEquilibriumCHV_bisect('bwd',P,T,Mp,Ms,**kwargs)

	# [Pd,Td,phip,phis,phip0,phis0] = binaryPhaseEquilibriumCHV(P,T,Mp,Ms,**kwargs)

	#MIXTURE PARAMETERS.
	if Kier or Hassan:
		vhm = delta*vhs
		alpha_p = alpha_p0*vhp/vhm
		alpha_s = alpha_s0*vhs/vhm
	elif Hassan_Var_Vol:
		Rtilde=phis+phip
		vhm=(phis/Rtilde)*vhs+(phip/Rtilde)*vhp
		alpha_p = alpha_p0*vhp/vhm				#Hassan: v_h is volume of hole in mixture or pure solvent
		alpha_s = alpha_s0*vhs/vhm	

	print 'P=',P,'T=',T,'phip=',phip,'phis=',phis
	#CALCULATION OF SOLVENT SOLUBILITY (MASS FRACTION) AT P, T.
	Xs = (Ms*phis/alpha_s)/(Mp*phip/alpha_p+Ms*phis/alpha_s)		#Kier Formula
	# Xs = (Ms*phis/alpha_s0)/(Mp*phip/alpha_p0+Ms*phis/alpha_s0)		#Condo Formula
	
	#CALCULATION OF VOLUME SWELLING (Vp/Vm) IN THE MIXTURE AT P, T.
	Sw = phip0/phip
	
	#PRINTING OF RESULTS OF SOLUBILITY AND SWELLING.
	print('At P = {}, T = {}, zeta = {}, delta = {};'.format(P,T,zeta,delta))
	print('Xs = {}, Sw = {};'.format(Xs,Sw))
	print('phip = {}, phis = {}, phip0 = {}, phis0 = {};'.format(phip,phis,phip0,phis0))

	Rtilde=phip+phis

	return [P,T,Xs,Sw,phip,phis,Rtilde,phip0,phis0]

def binarySolubilitySwellingCondo_Original(P,T,Mp,Ms,**kwargs):

	# Xs:the total solubility m of solvent and co-solvent.
	# Sw:the volume swelling.

	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))

	#PURE FLUID PARAMETERS.
	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar
	alpha_p0 = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_s0 = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	
	# CALCULATION OF VOLUME FRACTIONS AT P, T.

	if forward:
		[cphis,cphip,Rtilde,Rstilde,Rptilde]=binaryPhaseEquilibriumCondo_Original_bisect('fwd',P,T,Mp,Ms,**kwargs)
	if backward:
		[cphis,cphip,Rtilde,Rstilde,Rptilde]=binaryPhaseEquilibriumCondo_Original_bisect('bwd',P,T,Mp,Ms,**kwargs)

	kphis=cphis*Rtilde
	kphip=cphip*Rtilde
	kphip0=Rptilde
	kphis0=Rstilde


	#CALCULATION OF SOLVENT SOLUBILITY (MASS FRACTION) AT P, T.
	Xs = (Ms*kphis/alpha_s0)/(Mp*kphip/alpha_p0+Ms*kphis/alpha_s0)
	# Xs_wrong = (kphis*Rsstar)/(kphis*Rsstar+kphip*Rpstar)		#Not working
	# vhm=cphis*vhs+cphip*vhp
	# alpha_s=alpha_s0*vhs/vhm
	# alpha_p=alpha_p0*vhp/vhm
	# Xs_wrong = (Ms*kphis/alpha_s)/(Mp*kphip/alpha_p+Ms*kphis/alpha_s) #Not working

	#CALCULATION OF VOLUME SWELLING (Vp/Vm) IN THE MIXTURE AT P, T.
	Sw = kphip0/(kphip)
	
	#PRINTING OF RESULTS OF SOLUBILITY AND SWELLING.
	print('At P = {}, T = {}, zeta = {};'.format(P,T,zeta))
	print('Xs = {}, Sw = {};'.format(Xs,Sw))
	print('kphip = {}, kphis = {}, kphip0 = {}, kphis0 = {};'.format(kphip,kphis,kphip0,kphis0))

	return [P,T,Xs,Sw,kphip,kphis,Rtilde,kphip0,kphis0]

#####################################################################################################
#####################################################################################################

def calculateThermodynamicVariables(P0,T0,phi_p,phi_s,phi_p0,phi_s0,Mp,Ms,**kwargs):

	if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
		TD = ThermodynamicVariables(P0,T0,phi_p,phi_s,phi_p0,Mp,Ms,**kwargs)
		result = TD
		# print 'P=',P0,'T=',T0,'phi_s0=',phi_s0,'phi_p0=',phi_p0
		# prRed('P={},T={},phis0={},phip0={}'.format(P0,T0,phi_s0,phi_p0))

	elif not isListOrNpyArray(T0) and isListOrNpyArray(P0):
		result = [[range(0,len(P0))] for x in range(4)]
		T = range(0,len(P0))
		S_1 = range(0,len(P0))
		S_2 = range(0,len(P0))
		
		for i in range(0,len(P0)):
			TD = ThermodynamicVariables(P0[i],T0,phi_p[i],phi_s[i],phi_p0[i],Mp,Ms,**kwargs)
			T[i] = T0
			S_1[i] = TD[0]
			S_2[i] = TD[1]
			# print 'P=',P0[i],'T=',T0,'phi_s0=',phi_s0[i],'phi_p0=',phi_p0[i]
			# prRed('P={},T={},phis0={},phip0={}'.format(P0[i],T0,phi_s0[i],phi_p0[i]))

		result[0] = P0
		result[1] = T
		result[2] = S_1
		result[3] = S_2

	elif not isListOrNpyArray(P0) and isListOrNpyArray(T0):
		result = [[range(0,len(T0))] for x in range(4)]
		P = range(0,len(T0))
		S_1 = range(0,len(T0))
		S_2 = range(0,len(T0))

		for i in range(0,len(T0)):
			TD = ThermodynamicVariables(P0,T0[i],phi_p[i],phi_s[i],phi_p0[i],Mp,Ms,**kwargs)
			P[i] = P0
			S_1[i] = TD[0]
			S_2[i] = TD[1]
			# print 'P=',P0,'T=',T0[i],'phi_s0=',phi_s0[i],'phi_p0=',phi_p0[i]
			# prRed('P={},T={},phis0={},phip0={}'.format(P0,T0[i],phi_s0[i],phi_p0[i]))

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
			TD = ThermodynamicVariables(P0[i],T0[i],phi_p[i],phi_s[i],phi_p0[i],Mp,Ms,**kwargs)
			S_1[i] = TD[0]
			S_2[i] = TD[1]
			# print 'P=',P0[i],'T=',T0[i],'phi_s0=',phi_s0[i],'phi_p0=',phi_p0[i]
			# prRed('P={},T={},phis0={},phip0={}'.format(P0[i],T0[i],phi_s0[i],phi_p0[i]))

		result[0] = P0
		result[1] = T0
		result[2] = S_1
		result[3] = S_2
	
	else:
		raise ValueError('In calculateThermodynamicVariables: Unknown error involving P0 and T0.')

	return result

def calculateCondoThermodynamicVariables_Original(P0,T0,kphi_p,kphi_s,kphi_p0,kphi_s0,Mp,Ms,**kwargs):

	if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
		Rtilde=kphi_p+kphi_s
		cphi_s=kphi_s/Rtilde
		TD = CondoThermodynamicVariables_Original(P0,T0,Rtilde,cphi_s,Mp,Ms,**kwargs)
		result = TD
	
	elif not isListOrNpyArray(T0) and isListOrNpyArray(P0):
		result = [[range(0,len(P0))] for x in range(4)]
		T = range(0,len(P0))
		S_1 = range(0,len(P0))
		S_2 = range(0,len(P0))
		Rtilde = range(0,len(P0))
		cphi_s = range(0,len(P0))
	
		for i in range(0,len(P0)):

			Rtilde[i]=kphi_p[i]+kphi_s[i]
			cphi_s[i]=kphi_s[i]/Rtilde[i]
			TD = CondoThermodynamicVariables_Original(P0[i],T0,Rtilde[i],cphi_s[i],Mp,Ms,**kwargs)
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
		Rtilde = range(0,len(T0))
		cphi_s = range(0,len(T0))
		# print Rtilde
		# print T0

		for i in range(0,len(T0)):
			Rtilde[i]=kphi_p[i]+kphi_s[i]
			cphi_s[i]=kphi_s[i]/Rtilde[i]
			TD = CondoThermodynamicVariables_Original(P0,T0[i],Rtilde[i],cphi_s[i],Mp,Ms,**kwargs)
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
		Rtilde = range(0,len(T0))
		cphi_s = range(0,len(T0))

		for i in range(0,len(T0)):
			Rtilde[i]=kphi_p[i]+kphi_s[i]
			cphi_s[i]=kphi_s[i]/Rtilde[i]
			TD = CondoThermodynamicVariables_Original(P0[i],T0[i],Rtilde[i],cphi_s[i],Mp,Ms,**kwargs)
			S_1[i] = TD[0]
			S_2[i] = TD[1]

		result[0] = P0
		result[1] = T0
		result[2] = S_1
		result[3] = S_2
	
	else:
		raise ValueError('In calculateCondoThermodynamicVariables_Original: Unknown error involving P0 and T0.')

	return result

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

#####################################################################################################
#Experimental Solubility Correction Equation:
#####################################################################################################

def TaitVolume(P,T,v_0,alpha,B0,B1):
	C = 0.0894			#Universal Constant
	B_T = B0*math.exp(-B1*T)	#B(T): Tait Parameter
	v_0_T = v_0*math.exp(alpha*T)		#Zero Pressure Specific Volume v(0,T), alpha: Thermal Expansion Coefficient
	v = v_0_T*(1-(C*math.log(1+(P/B_T))))
	return v

def calculateTaitVolume(P0,T0,v_0,alpha,B0,B1):
    
  if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
      v = TaitVolume(P0,T0,v_0,alpha,B0,B1)
      result = v
    
  elif not isListOrNpyArray(P0):
      result = [[range(0,len(T0))] for x in range(2)]
      v = range(0,len(T0))
      for i in range(0,len(T0)):
          v[i] = TaitVolume(P0,T0[i],v_0,alpha,B0,B1)
      result[0] = v
      result[1] = T0
    
  elif not isListOrNpyArray(T0):
      result = [[range(0,len(P0))] for x in range(2)]
      v = range(0,len(P0))
      for i in range(0,len(P0)):
          v[i] = TaitVolume(P0[i],T0,v_0,alpha,B0,B1)
      result[0] = v
      result[1] = P0

  elif isListOrNpyArray(P0) and isListOrNpyArray(T0):
      result = [[range(0,len(P0))] for x in range(2)]
      v = range(0,len(P0))
      for i in range(0,len(P0)):
          v[i] = TaitVolume(P0[i],T0[i],v_0,alpha,B0,B1)
      result[0] = v
      result[1] = P0

  return result

def ExpSwellingDataCorrectedSolubility(P0,T0,X0_total,Sw0_total,Rgas,vppure):

	#Hassan Forumulas for Total Solubility below:
	# vpmix = Sw0_total*(1-X0_total)*vppure				#Alternate valid formula. All are experimental values. vppure is from tait equation.
	# vs = (vpmix/(1-X0_total))-vppure				#This is correct.
	#Incorrect Formula ==> X0_total_corrected_hassan = X0_total + (Rgas*vs)

	#Chul Park Forumulas for Fraction Solubility below:
	X0_fraction = X0_total/(1-X0_total)
	vpmix = (Sw0_total/(1+X0_fraction))*vppure			#All are experimental values. vppure is from tait equation.
	vs = (((1+X0_fraction)*vpmix)-vppure)
	X0_fraction_corrected = X0_fraction + (Rgas*vs)
	X0_total_corrected_park = X0_fraction_corrected/(1+X0_fraction_corrected)
	
	X0_total_corrected_hassan = 0.0			#It turns out that my formula is wrong.
	return X0_total_corrected_hassan,X0_total_corrected_park

def calculateExpSwellingDataCorrectedSolubility(P0,T0,X0_total,Sw0_total,Rgas,vppure):

  if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
      X0_total_corrected_hassan, X0_total_corrected_park = ExpSwellingDataCorrectedSolubility(P0,T0,X0_total,Sw0_total,Rgas,vppure)
      result = X0_total_corrected_hassan, X0_total_corrected_park
    
  elif not isListOrNpyArray(P0):
      result = [[range(0,len(T0))] for x in range(3)]
      X0_total_corrected_hassan = range(0,len(P0))
      X0_total_corrected_park = range(0,len(P0))
      for i in range(0,len(T0)):
          X0_total_corrected_hassan[i],X0_total_corrected_park[i] = ExpSwellingDataCorrectedSolubility(P0,T0[i],X0_total[i],Sw0_total[i],Rgas[i],vppure[i])
      result[0] = X0_total_corrected_hassan
      result[1] = X0_total_corrected_park
      result[2] = T0
    
  elif not isListOrNpyArray(T0):
      result = [[range(0,len(P0))] for x in range(3)]
      X0_total_corrected_hassan = range(0,len(P0))
      X0_total_corrected_park = range(0,len(P0))
      for i in range(0,len(P0)):
          X0_total_corrected_hassan[i],X0_total_corrected_park[i] = ExpSwellingDataCorrectedSolubility(P0[i],T0,X0_total[i],Sw0_total[i],Rgas[i],vppure[i])
      result[0] = X0_total_corrected_hassan
      result[1] = X0_total_corrected_park
      result[2] = P0

  elif isListOrNpyArray(P0) and isListOrNpyArray(T0):
      result = [[range(0,len(P0))] for x in range(3)]
      X0_total_corrected_hassan = range(0,len(P0))
      X0_total_corrected_park = range(0,len(P0))
      for i in range(0,len(P0)):
          X0_total_corrected_hassan[i],X0_total_corrected_park[i] = ExpSwellingDataCorrectedSolubility(P0[i],T0[i],X0_total[i],Sw0_total[i],Rgas[i],vppure[i])
      result[0] = X0_total_corrected_hassan
      result[1] = X0_total_corrected_park
      result[2] = P0

  return result

def CorrectSolubilityExp(P0,T0,X0_total,Rgas,vppure,X_total_theory,Sw_total_theory,phip,phis,Mp,Ms,**kwargs):

	#Input kwargs should be:
	# kwargs = {'zeta':zeta,'delta':delta,'Mp':Mp,'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar,'Ms':Ms,'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar,'Kier':Kier,'Hassan':Hassan,'Hassan_Var_Vol':Hassan_Var_Vol,'Condo':Condo}

	# for key,value in kwargs.items():
	# 	exec "%s=%s" % (key,value)
	
	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))

	vhp = kB*Tpstar/Ppstar
	vhs = kB*Tsstar/Psstar
	alpha_p0 = (Ppstar*Mp)/(kB*Tpstar*Rpstar)
	alpha_s0 = (Psstar*Ms)/(kB*Tsstar*Rsstar)
	
	#MIXTURE PARAMETERS.
	if Kier or Hassan:
		vhm = delta*vhs
		alpha_p = alpha_p0*vhp/vhm
		alpha_s = alpha_s0*vhs/vhm
	elif Hassan_Var_Vol:
		Rtilde=phis+phip
		vhm=(phis/Rtilde)*vhs+(phip/Rtilde)*vhp
		alpha_p = alpha_p0*vhp/vhm				#Hassan: v_h is volume of hole in mixture or pure solvent
		alpha_s = alpha_s0*vhs/vhm	

	# vpmix = (vhm)/((phip/(alpha_p/Mp))+(phis/(alpha_s/Ms)))
	# print 'first vpmix\t:', vpmix
	vpmix = (1-X_total_theory)*((vhm*(alpha_p/Mp))/(phip)) 				#X_total_theory is total X.
	# print 'second vpmix\t:', vpmix
	#Use of vppure from tait equation in below two vpmix formulas will result in slight different result from above two.
	# If you use vppure from SL theory then the results will be exact. I think vppure from SL EOS is much consistent.  
	# vpmix = Sw_total_theory*(1-X_total_theory)*vppure					#Alternate valid formula. All are experimental values. vppure is from tait equation.
	# print 'third vpmix\t:', vpmix
	# X_fraction_theory = X_total_theory/(1-X_total_theory)				#Convert Xtotal to Xfraction
	# vpmix = (Sw_total_theory/(1+X_fraction_theory))*vppure				#All are experimental values. vppure is from tait equation.
	# print 'fourth vpmix\t:', vpmix

	# print 'first Sw_total_theory\t:', Sw_total_theory
	# Sw_total_theory = vpmix/(vppure*(1-X_total_theory))				#Correct. Swelling Alternate Formula. Giving exact same result as input Sw to the function. 
	# print 'second Sw_total_theory\t:', Sw_total_theory

	#Hassan Forumulas for Total Solubility below:
	# vs = (vpmix/(1-X_total_theory))-vppure				#This is correct.
	# print 'first vs\t:', vs
	#Incorrect formula ===> X0_total_corrected = X0_total + (Rgas*vs)*(1-X_total_theory)
	# print 'first X0_total_corrected\t:', X0_total_corrected

	#Chul Park Forumulas for Fraction Solubility below:
	X_fraction_theory = X_total_theory/(1-X_total_theory)				#Convert Xtotal to Xfraction
	X0_fraction = X0_total/(1-X0_total)									#Convert Xtotal to Xfraction
	vs = (((1+X_fraction_theory)*vpmix)-vppure)
	# print 'second vs\t:', vs
	X0_fraction_corrected = X0_fraction + (Rgas*vs)

	X0_total_corrected = X0_fraction_corrected/(1+X0_fraction_corrected)#Convert from Xfraction to Xtotal
	# print 'second X0_total_corrected\t:', X0_total_corrected
	# print 'Only X0_fraction_corrected\t:', X0_fraction_corrected

	return X0_total_corrected

def calculateCorrectSolubilityExp(P0,T0,X0_total,Rgas,vppure,X_total_theory,Sw_total_theory,phip,phis,Mp,Ms,**kwargs):

  if not isListOrNpyArray(P0) and not isListOrNpyArray(T0):
      X0_total_corrected = CorrectSolubilityExp(P0,T0,X0_total,Rgas,vppure,X_total_theory,Sw_total_theory,phip,phis,Mp,Ms,**kwargs)
      result = X0_total_corrected
    
  elif not isListOrNpyArray(P0):
      result = [[range(0,len(T0))] for x in range(2)]
      X0_total_corrected = range(0,len(T0))
      for i in range(0,len(T0)):
          X0_total_corrected[i] = CorrectSolubilityExp(P0,T0[i],X0_total[i],Rgas[i],vppure[i],X_total_theory[i],Sw_total_theory[i],phip[i],phis[i],Mp,Ms,**kwargs)
      result[0] = X0_total_corrected
      result[1] = T0
    
  elif not isListOrNpyArray(T0):
      result = [[range(0,len(P0))] for x in range(2)]
      X0_total_corrected = range(0,len(P0))
      for i in range(0,len(P0)):
          X0_total_corrected[i] = CorrectSolubilityExp(P0[i],T0,X0_total[i],Rgas[i],vppure[i],X_total_theory[i],Sw_total_theory[i],phip[i],phis[i],Mp,Ms,**kwargs)
      result[0] = X0_total_corrected
      result[1] = P0

  elif isListOrNpyArray(P0) and isListOrNpyArray(T0):
      result = [[range(0,len(P0))] for x in range(2)]
      X0_total_corrected = range(0,len(P0))
      for i in range(0,len(P0)):
          X0_total_corrected[i] = CorrectSolubilityExp(P0[i],T0[i],X0_total[i],Rgas[i],vppure[i],X_total_theory[i],Sw_total_theory[i],phip[i],phis[i],Mp,Ms,**kwargs)
      result[0] = X0_total_corrected
      result[1] = P0

  return result

#####################################################################################################
#Self Regression of Zeta and Delta to fit on Solubility:
#####################################################################################################

def residualFunction_SelfFit(A0,A,weight=1.0):
	# print 'weight is', weight

	if len(A0) != len(A):
		raise ValueError('In residual: The number of experimental points and number of theoretical points are not equal.')
	
	residual = npy.zeros(len(A0))

	for i in range(0,len(A0)):
		residual[i] = weight*((A0[i]-A[i]))/A0[i]  #Kier original had no hash and no absolute	
	
	return residual

def binaryResidualCorrected_Xs(theory,Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,**kwargs):

	m_s = npy.zeros(len(P0_X))
	Sw = npy.zeros(len(P0_S))
	res_X = npy.zeros(len(P0_X))
	res_S = npy.zeros(len(P0_S))
	
	if len(P0_X) != len(P0_S):
		warnings.warn('In binaryResidualCorrected_Xs: Mismatch in solubility and swelling data number. Results may be skewed.')

	# suppress_print = kwargs.pop('suppress_print',False)
	# if not suppress_print:
	
	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))

	# for key,value in kwargs.items():
	# 	print '%s=%s' % (key,value)

	# for key,value in kwargs.items():
	# 	exec "%s=%s" % (key,value)
	
	if 'X' in fit_type:

		result = calculateBinarySolubilitySwelling(theory,P0_X,T0_X,Mp,Ms,**kwargs)
		m_s = result[2]
		phip = result[4]
		phis = result[5]
		answer= calculateCorrectSolubilityExp(P0_X,T0_X,X0_X,Rgas,vppure,m_s,phip,phis,Mp,Ms,**kwargs)
		m_s_corrected = answer[0]
		res_X = residualFunction_SelfFit(m_s_corrected,m_s,1.0-fs)
	if 'S' in fit_type:
		result = calculateBinarySolubilitySwelling(theory,P0_S,T0_S,Mp,Ms,**kwargs)
		Sw = result[3]
		res_S = residualFunction_SelfFit(S0_S,Sw,fs)
	
	if 'X' in fit_type and 'S' in fit_type:
		residual = npy.concatenate((res_X,res_S),axis=0)
	elif 'X' in fit_type:
		residual = res_X
	elif 'S' in fit_type:
		residual = res_S
	else:
		raise ValueError('In binaryResidualCorrected_Xs: fit_type must contain X and/or S.')

	return residual

def precise_delta_by_SumResidual_at_given_zeta(Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,min_delta,max_delta,num_of_delta_divided,num_of_precision_loops_for_delta,allowed_error_in_delta,**kwargs):

	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))

	delta_array = npy.linspace(min_delta, max_delta, num=num_of_delta_divided)
	sumRes_array = npy.ones(num_of_delta_divided)
	SSE_array = npy.ones(num_of_delta_divided)

	succcess = 0
	# Minimum_delta_found = 0

	for j in range(len(delta_array)):
		delta = delta_array[j]
		isdeltaFlop = 0
		kwargs = {'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar,'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar,'zeta':zeta,'delta':delta,'method':'disparate','verbose':False,'Kier':Kier,'Hassan':Hassan,'Condo':Condo,'Hassan_Var_Vol':Hassan_Var_Vol,'forward':forward,'backward':backward}
		# res = npy.ones(len(X0_X))
		try:
			res = binaryResidualCorrected_Xs('CHV',Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,**kwargs)
		except:
			isdeltaFlop = 1
			res = -999*npy.ones(len(X0_X))
		res_array = npy.array(res)
		# prGreen('residual delta array = {}'.format(res_array))

		sumRes_array[j] = npy.sum(res_array)
		SSE_array[j] = npy.sqrt(npy.sum(npy.power(res_array, 2))/len(res_array))

		# prGreen('sumRes of residual delta array = {}'.format(npy.sum(res_array)))
		prGreen('At non-precise delta: {}, returned sumRes is:'.format(delta))
		prGreen('{}'.format(npy.sum(npy.sum(res_array))))

		if sumRes_array[j] < 0 and isdeltaFlop==0:
			success = 1
		elif sumRes_array[j] > 0 and isdeltaFlop==0 and success == 0:
			raise ValueError('In Sum of Residual Array is Positive at the Suggested Minimum Delta. Decrease the value of Suggested Minimum Delta')

		if sumRes_array[j] > 0 and j>0 and isdeltaFlop==0:
			delta_low = delta_array[j-1]
			sumRes_low = sumRes_array[j-1]
			SSE_low = SSE_array[j-1]
		
			delta_high = delta_array[j]
			sumRes_high = sumRes_array[j]
			SSE_high = SSE_array[j]

			for k in range(num_of_precision_loops_for_delta):
				prGreen('Delta Precising Loop # {}'.format(k))
				delta_trail = (delta_low+delta_high)/2
				kwargs = {'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar,'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar,'zeta':zeta,'delta':delta_trail,'method':'disparate','verbose':False,'Kier':Kier,'Hassan':Hassan,'Condo':Condo,'Hassan_Var_Vol':Hassan_Var_Vol,'forward':forward,'backward':backward}
				res = binaryResidualCorrected_Xs('CHV',Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,**kwargs)
				res_array = npy.array(res)
				sumRes_trail = npy.sum(res_array)
				SSE_trail = npy.sqrt(npy.sum(npy.power(res_array, 2))/len(res_array))

				# prGreen('Precision Residual = {}'.format(res_array))
				# prGreen('Precision sumRes of residual array = {}'.format(sumRes_trail))

				prGreen('At precision delta_trail: {}, return delta sumRes_trail is:'.format(delta_trail))
				prGreen('{}'.format(npy.sum(sumRes_trail)))

				if sumRes_trail<0:
					# delta_high = delta_high
					# sumRes_high = sumRes_high
					delta_low = delta_trail
					sumRes_low = sumRes_trail
					SSE_low = SSE_trail

				elif sumRes_trail>0:
					# delta_low = delta_low
					# sumRes_low = sumRes_low
					delta_high = delta_trail
					sumRes_high = sumRes_trail
					SSE_high = SSE_trail
				
				if abs(delta_high-delta_low)<=allowed_error_in_delta:
					# prGreen('Required delta precision achieved at loop # {}'.format(k))
					prGreen('Required delta precision achieved at delta loop # {} and the value of precise delta is:'.format(k))
					prGreen('{}'.format(delta_low))

					break
			
			# prGreen('Minimum delta found')
			# Minimum_delta_found = 1
			delta_precise =  delta_low
			sumRes_precise =  sumRes_low
			SSE_precise = SSE_low
			prGreen('Error minimization successful through delta precision. The value of precise delta is: {}'.format(delta_precise))
			break

	# if succcess == 0:
	# 	prPurple(success)
	# 	delta_precise =  npy.nan
	# 	sumRes_precise =  npy.nan
	# 	raise ValueError('delta minimum not achieved in the given range. Increase the suggested value of max_delta')

	# print 'sumRes_precise for delta is ready to be returned = ', sumRes_precise
	# print sumRes_low, sumRes_high, sumRes_trail

	return [delta_precise,sumRes_precise,delta_high,sumRes_high,delta_array,sumRes_array,SSE_precise,SSE_high,SSE_array]

def precise_zeta_SumResidual(Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,min_zeta,max_zeta,num_of_zeta_divided,num_of_precision_loops_for_zeta,allowed_error_in_zeta,min_delta,max_delta,num_of_delta_divided,num_of_precision_loops_for_delta,allowed_error_in_delta,**kwargs):

	for key,value in kwargs.iteritems():
		exec('{} = value'.format(key))

	zeta_array = npy.linspace(min_zeta, max_zeta, num=num_of_zeta_divided)
	delta_array_of_best_fit_at_given_zetas = npy.ones(num_of_zeta_divided)
	sumRes_array_zeta = npy.ones(num_of_zeta_divided)
	SSE_array_zeta = npy.ones(num_of_zeta_divided)

	succcess = 0
	Minimum_zeta_found = 0

	for i in range(len(zeta_array)):

		zeta = zeta_array[i]
		iszetaFlop = 0
		kwargs = {'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar,'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar,'zeta':zeta,'method':'disparate','verbose':False,'Kier':Kier,'Hassan':Hassan,'Condo':Condo,'Hassan_Var_Vol':Hassan_Var_Vol,'forward':forward,'backward':backward}
		result = precise_delta_by_SumResidual_at_given_zeta(Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,min_delta,max_delta,num_of_delta_divided,num_of_precision_loops_for_delta,allowed_error_in_delta,**kwargs)
		delta_precise = result[0]
		sumRes_precise = result[1]
		delta_high = result[2]
		sumRes_high = result[3]
		delta_array = result[4]
		sumRes_array_delta = result[5]
		SSE_precise = result[6]
		SSE_high = result[7]
		SSE_array_delta = result[8]

		delta_array_of_best_fit_at_given_zetas[i] = delta_precise
		sumRes_array_zeta[i] = sumRes_precise
		SSE_array_zeta[i] = SSE_precise

		if math.isnan(sumRes_precise):
			iszetaFlop = 1
			# res = -999*res
		
		# res_array = npy.array(res)
		# prRed('residual = {}'.format(res_array))

		# prRed('At non-precise zeta: returned delta residual array is: {}'.format(npy.sum(sumRes_array_delta)))
		prRed('At non-precise zeta: {},'.format(zeta))
		prRed('return delta sumRes is :{}'.format(npy.sum(sumRes_precise)))
		prRed('and return delta SSE is :{}'.format(npy.sum(SSE_precise)))

		prRed('Final sumRes array at precise deltas and non-precise zetas is: {},'.format(sumRes_array_zeta))
		prRed('with final SSE array at precise deltas and non-precise zetas is: {},'.format(SSE_array_zeta))
		prRed('where non-precise zeta array is: {},'.format(zeta_array))
		prRed('and precise delta array is: {},'.format(delta_array_of_best_fit_at_given_zetas))

		# print sumRes_array_zeta[i], sumRes_array_zeta[i-1], iszetaFlop

		if sumRes_array_zeta[i] < 0 and iszetaFlop==0:
			success = 1
		elif sumRes_array_zeta[i] > 0 and iszetaFlop==0 and success == 0:
			raise ValueError('delta range is not adequate at given range of Zeta. Change the range of delta or zeta')

		if abs(SSE_array_zeta[i]) > abs(SSE_array_zeta[i-1]) and i>0 and iszetaFlop==0:
			zeta_low = zeta_array[i-1]
			sumRes_low_zeta = sumRes_array_zeta[i-1]
			SSE_low_zeta = SSE_array_zeta[i-1]
			zeta_high = zeta_array[i]
			sumRes_high_zeta = sumRes_array_zeta[i]
			SSE_high_zeta = SSE_array_zeta[i]
			for k in range(num_of_precision_loops_for_zeta):
				prRed('Zeta Precising Loop # {}'.format(k))
				zeta_trail = (zeta_low+zeta_high)/2
				kwargs = {'Ppstar':Ppstar,'Tpstar':Tpstar,'Rpstar':Rpstar,'Psstar':Psstar,'Tsstar':Tsstar,'Rsstar':Rsstar,'zeta':zeta_trail,'method':'disparate','verbose':False,'Kier':Kier,'Hassan':Hassan,'Condo':Condo,'Hassan_Var_Vol':Hassan_Var_Vol,'forward':forward,'backward':backward}
				result = precise_delta_by_SumResidual_at_given_zeta(Rgas,vppure,P0_X,T0_X,X0_X,P0_S,T0_S,S0_S,Mp,Ms,fit_type,fs,min_delta,max_delta,num_of_delta_divided,num_of_precision_loops_for_delta,allowed_error_in_delta,**kwargs)
				delta_precise_trail = result[0]
				sumRes_precise_trail = result[1]
				delta_high_trail = result[2]
				sumRes_high_trail = result[3]
				delta_array_trail = result[4]
				sumRes_array_delta_trail = result[5]
				SSE_precise_trail = result[6]
				SSE_high_trail = result[7]
				SSE_array_delta_trail = result[8]

				prRed('At precision zeta_trail: {},'.format(zeta_trail))
				prRed(' return delta sumRes_trail is:{},'.format(npy.sum(sumRes_precise_trail)))
				prRed(' return delta SSE_trail is:{},'.format(SSE_precise_trail))

				# print 'Precision sumRes of residual array = ',sumRes_precise_trail
				# prRed('Precision sumRes of residual array = {}'.format(sumRes_precise_trail))

				if abs(SSE_precise_trail)<abs(SSE_low_zeta):
					# zeta_high = zeta_high
					# sumRes_high_delta = sumRes_high_delta
					zeta_low = zeta_trail
					sumRes_low_zeta = sumRes_precise_trail
					SSE_low_zeta = SSE_precise_trail
					delta_low=delta_precise_trail
					SSE_high=SSE_high_trail

					# delta_array=delta_array_trail
					# sumRes_array_delta=sumRes_array_delta_trail
					# SSE_array_delta=SSE_array_delta_trail


				elif abs(SSE_precise_trail)>abs(SSE_low_zeta):
					# zeta_low = zeta_low
					# sumRes_low_zeta = sumRes_low_zeta
					zeta_high = zeta_trail
					sumRes_high_zeta = sumRes_precise_trail
					SSE_high_zeta = SSE_precise_trail
					
					delta_high=delta_high_trail
					sumRes_high=sumRes_high_trail
					
				if abs(zeta_high-zeta_low)<=allowed_error_in_zeta:
					prRed('Required zeta precision achieved at zeta loop # {} and the value of precise zeta is:'.format(k))
					prRed('{}'.format(zeta_low))

					break
			
			# Minimum_zeta_found = 1
			zeta_precise =  zeta_low
			sumRes_precise =  sumRes_low_zeta
			SSE_precise =  SSE_low_zeta
			delta_precise =  delta_low

			prRed('Error minimization successful through zeta precision. The value of precise zeta is: {}'.format(zeta_precise))
			break

	# if succcess == 0:
	# 	zeta_precise =  npy.nan
	# 	sumRes_precise =  npy.nan
	# 	raise ValueError('zeta minimum not achieved in the given range. Increase the suggested value of max_zeta')

	return [zeta_precise,delta_precise,sumRes_precise,zeta_high,delta_high,sumRes_high,zeta_array,delta_array_of_best_fit_at_given_zetas,sumRes_array_zeta,SSE_precise,SSE_high,SSE_array_zeta]
