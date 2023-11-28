#Sanchez-Lacombe parameters for the pure POLYMERS.

#======================================================
#Characteristic Parameters
#======================================================
# Pstar = [MPa]
# Tstar = [K]
# Rstar = [g/cm3]

#############################################################################################
# Polymer_Type='PMMA' #PS or PMMA or DME or LPP or BPP or PLA or LDPE
# Solvent='CO2' 
#############################################################################################

def Tait_Parameters_of_Different_Polymers(**kwargs):
  
  for key,value in kwargs.items():
    exec "%s='%s'" % (key,value)

  print 'Polymer Type', Polymer_Type, 'Solvent', Solvent, 'PVT and Tg Data taken from', Parameters_Paper

  #################################################################################
  #PLA (4060D) Values Regressed By Park:
  #################################################################################

  if Polymer_Type=='PLA' and Parameters_Paper=='Self_Park':
    # Data Paper: Determination of carbon dioxide solubility in polylactide acid with accurate PVT properties
    # Author: Park et al. (2014)
    v_0 =    0.61401476 #+/- 0.00666235 (1.09%) (init = 0.6)   
    alpha =  8.4676e-04 #+/- 2.7302e-05 (3.22%) (init = 0.0003)
    B0 =     321.437165 #+/- 120.421606 (37.46%) (init = 16000)
    B1 =     0.00327061 #+/- 9.2069e-04 (28.15%) (init = 0.005)

  if Polymer_Type=='PMMA' and Parameters_Paper=='Self_Grassia':

    v_0 =    0.69092811 #+/- 0.00200602 (0.29%) (init = 0.6)
    alpha =  5.9946e-04 #+/- 7.3100e-06 (1.22%) (init = 0.0003)
    B0 =     569.606378 #+/- 56.0986237 (9.85%) (init = 16000)
    B1 =     0.00305138 #+/- 2.4243e-04 (7.94%) (init = 0.005)

  if Polymer_Type=='PS' and Parameters_Paper=='Self_Grassia':
    v_0 =    0.78163187 #+/- 0.00178507 (0.23%) (init = 0.6)
    alpha =  5.8637e-04 #+/- 5.6814e-06 (0.97%) (init = 0.0003)
    B0 =     462.943089 #+/- 29.0677854 (6.28%) (init = 16000)
    B1 =     0.00300811 #/- 1.5205e-04 (5.05%) (init = 0.005)
  
  return v_0,alpha,B0,B1
