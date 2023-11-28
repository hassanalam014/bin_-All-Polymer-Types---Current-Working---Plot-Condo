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

P_atm = 0.101325
M_infinity = 9E9

def Parameters_of_Different_Polymers(**kwargs):
	
    for key,value in kwargs.items():
		exec "%s='%s'" % (key,value)
    
    print 'Polymer Type', Polymer_Type, 'Solvent', Solvent, 'PVT and Tg Data taken from', Parameters_Paper

    #################################################################################
    #PLA (4060D) Values Regressed By Park:
    #################################################################################

    if Polymer_Type=='PLA' and Parameters_Paper=='Self_Park':
        # Paper: Determination of carbon dioxide solubility in polylactide acid with accurate PVT properties
        # Author: Park et al. (2014)
        Ppstar =  464.127112 #+/- 13.7891457 (2.97%) (init = 572.96)
        Tpstar =  566.767133 #+/- 2.72170304 (0.48%) (init = 591.56)
        Rpstar =  1.37495743 #+/- 0.00161557 (0.12%) (init = 1.364)
        Mp=190000
        P_exp = []
        Tg_exp = []
        # Ppstar_Park=572.96
        # Tpstar_Park=591.56
        # Vpstar_Park=0.73291
        # Rpstar_Park=1/Vpstar_Park

    #################################################################################
    #PMMA Values Regressed By Myself:
    #################################################################################

    if Polymer_Type=='PMMA' and Parameters_Paper=='Self_Schmidt':
        # Data 1 Very Good Matching 1
        # Paper: Good 2000 PVT of PMMA Schmidt ma991722h
        # Author: Schmidt et al. (2000)
        Ppstar=577.933210    #+/- 22.7932121 (3.94%)
        Tpstar=677.583131    #+/- 7.01760682 (1.04%)
        Rpstar=1.27392163    #+/- 0.00421348 (0.33%)
        Mp=75000

    if Polymer_Type=='PMMA' and Parameters_Paper=='Self_Grassia':
        # Data 2 Very Good Matching 2
        # Paper: Best 2011 PVT and Tg of PMMA Grassia 1-s2.0-S0022309310005338-main
        # Author: Grassia et al. (2011)
        Ppstar=562.397384    #+/- 7.52128331 (1.34%)
        Tpstar=654.343418    #+/- 2.72548523 (0.42%)
        Rpstar=1.28146781    #+/- 0.00153252 (0.12%)
        Mp=120000

    if Polymer_Type=='PMMA' and Parameters_Paper=='Self_Walsh':
        # Data 3 Overlap and Significant Matching but Not Complete Matching
        # Paper: Old 1992 PVT of PMMA PII_ 0032-3861(92)90694-R
        # Author: Walsh et al. (1992)
        Ppstar=564.823646    #+/- 17.9901923 (3.19%)
        Tpstar=667.892308    #+/- 5.07983058 (0.76%)
        Rpstar=1.28420005    #+/- 0.00323659 (0.25%)
        Mp = M_infinity

    if Polymer_Type=='PMMA' and Parameters_Paper=='Self_Wen':
        # Data 4 Neither Matching Nor Overlap
        # Paper: Good 2001 PVT and Tg of PMMA Wen ma010023d
        # Author: Wen (2001)
        Ppstar=479.377365    #+/- 31.3221097 (6.53%)
        Tpstar=709.595728    #+/- 7.99245188 (1.13%)
        Rpstar=1.28295203    #+/- 0.00451412 (0.35%)
        Mp=387000

    #################################################################################
    #PC Values Regressed By Myself:
    #################################################################################

    if Polymer_Type=='PC' and Parameters_Paper=='Self_Aravind':
        # Data 1 Matching 1
        # Paper: Best 2012 PVT and Tg of PC Aravind 1-s2.0-S014294181100153X-main
        # Author: Aravind et al. (2012)
        Ppstar=477.201600      #+/- 6.50120144 (1.36%) (init = 60)
        Tpstar=745.977589      #+/- 2.32835065 (0.31%) (init = 50)
        Rpstar=1.28039144      #+/- 0.00124800 (0.10%) (init = 2)
        Mp = M_infinity

    if Polymer_Type=='PC' and Parameters_Paper=='Self_Sato':
        # Data 2 Matching 2
        # Paper: Good 1997 PVT of PC (SICI)1097-4628(19971003)66_1_141__AID-APP17_3.0.CO;2-4
        # Author: Sato et al. (1997)
        Ppstar=500.834279      #+/- 11.7676771 (2.35%) (init = 60)
        Tpstar=751.670242      #+/- 4.73421735 (0.63%) (init = 50)
        Rpstar=1.28175322      #+/- 0.00261206 (0.20%) (init = 2)
        Mp=60000

    if Polymer_Type=='PC' and Parameters_Paper=='Self_Rudolph':
        # Data 3 Overlap But Not Matching
        # Paper: Best 2016 PVT and Tg of PC Rudolph Rudolph2016_Article_WLFModelForThePressureDependen
        # Author: Rudolph et al. (2016)
        Ppstar=378.30332       #+/- 11.5504425 (3.05%) (init = 60)
        Tpstar=625.52729       #+/- 3.99028295 (0.64%) (init = 50)
        Rpstar=1.33244470      #+/- 0.00373602 (0.28%) (init = 2)
        Mp=30500

    if Polymer_Type=='PC' and Parameters_Paper=='Self_Kikuchi':
        # Data 4 Neither Matching Nor Overlap
        # Paper: Good 2003 PVT and Tg of PS and PC Kikuchi Thermal Conductivity
        # Author: Kikuchi et al. (2003)
        Ppstar=899.363789      #+/- 14.6137301 (1.62%) (init = 60)
        Tpstar=659.224862      #+/- 2.94947589 (0.45%) (init = 50)
        Rpstar=1.22050074      #+/- 0.00213321 (0.17%) (init = 2)
        Mp=73700

    #################################################################################
    #PS Values Regressed By Myself:
    #################################################################################

    if Polymer_Type=='PS' and Parameters_Paper=='Self_Kier':
        # Data # 1: Matching But Not Exactly Kier Zoller Wash Data
        # Paper: NONE (Kier Original PVT Data)
        # Author: Zoller et al.    
        Ppstar=  430.896437      #+/- 9.16502716 (2.13%) (init = 60)
        Tpstar=  690.581024      #+/- 2.83119250 (0.41%) (init = 50)
        Rpstar=  1.11801998      #+/- 0.00128308 (0.11%) (init = 2)
        Mp=110000

    if Polymer_Type=='PS' and Parameters_Paper=='Self_Kikuchi':
        # Data#2: Overlap But Not Matching
        # Paper: Good 2003 PVT and Tg of PS and PC Kikuchi Thermal Conductivity
        # Author: Kikuchi (2003)
        Ppstar=  624.048947      #+/- 16.3388656 (2.62%) (init = 60)
        Tpstar=  623.692994      #+/- 4.00763066 (0.64%) (init = 50)
        Rpstar=  1.13205301      #+/- 0.00257298 (0.23%) (init = 2)
        Mp=25000

    if Polymer_Type=='PS' and Parameters_Paper=='Self_Park':
        # Data#3: Matching But Not Exactly 2004 Very few points
        # Paper: Good 2004 PVT of PS and PP adv.20020
        # Author: Park (2004)
        Ppstar=  265.473727      #+/- 24.9282071 (9.39%) (init = 275.8)
        Tpstar=  798.859724      #+/- 14.5314703 (1.82%) (init = 803.1)
        Rpstar=  1.07238654      #+/- 0.00508400 (0.47%) (init = 1.072)
        Mp=238000

    if Polymer_Type=='PS' and Parameters_Paper=='Self_Grassia':
        # Data#4: Matching But Not Exactly 2011 Best
        # Paper: Best 2011 PVT Tg of PS app.34789
        # Author: Grassia (2011)
        Ppstar=  462.343780      #+/- 6.99594722 (1.51%) (init = 60)
        Tpstar=  647.334764      #+/- 2.98020363 (0.46%) (init = 50)
        Rpstar=  1.14149853      #+/- 0.00157953 (0.14%) (init = 2)
        Mp=290000

    #################################################################################
    #Pre-Thesis: Values Directly Taken From Other's Papers:
    #################################################################################

    #PURE POLYMERS DATA:
    if Polymer_Type=='PS' and Parameters_Paper=='Condo':
        #Sanchez-Lacombe parameters for the pure POLYMER.
        # Condo PS Parameters:
        Mp = 196000.0
        Ppstar = 357.0
        Tpstar = 735.0
        Rpstar = 1.105
        # # Iterated Values From Pure PS Flexible System:
        # g=1.67 
        # epsilon_p=8013 
        # x=0.311

    if Polymer_Type=='PS' and Parameters_Paper=='Kier':

        # Iterated Values From Solubility of PS/CO2 System:
        Ppstar = 421.762455
        Tpstar = 687.788143
        Rpstar = 1.11768655


    if Polymer_Type=='PMMA' and Parameters_Paper=='Condo':
        #Sanchez-Lacombe parameters for the pure POLYMER.
        # Condo PMMA Parameters:
        Mp = 196000.0
        Ppstar = 503.0
        Tpstar = 696.0
        Rpstar = 1.269
        # Iterated Values From Pure PS Flexible System:
        # g=1.66 
        # epsilon_p=8094 
        # x=0.323

    if Polymer_Type=='PC' and Parameters_Paper=='Pre-Thesis':
        # For PC Ref: Zoller Paper, A Studey of PVT Relationships of Four Related Amorphous Polymers
        Mp = M_infinity
        Ppstar = 574.4             #From huge List of SL EOS Parameters
        Tpstar = 728.0             #From huge List of SL EOS Parameters
        Rpstar = 1.2925            #From huge List of SL EOS Parameters

    #PURE SOLVENT DATA:
    if Solvent=='CO2': #and Parameters_Paper=='Kier':
       #Sanchez-Lacombe parameters for the pure SOLVENT.
       #Kier Value of CO2
        Ms = 44.01
        Psstar = 419.915520
        Tsstar = 341.772507
        Rsstar = 1.39745988
        Vsstar = 0.71558405

    if Solvent=='CO2' and Parameters_Paper=='Condo':
        #Sanchez-Lacombe parameters for the pure SOLVENT.
        # Condo CO2 Parameters:
        Ms = 44.01
        Psstar = 574.0
        Tsstar = 308.64
        Rsstar = 1.505

    #Experimental Data of Binary Mixtures:
    if Polymer_Type=='PS' and Solvent=='CO2':

        #Experiment Data PS;
        Tg_exp=[373.0,328.3,308.2,305.0,304.5]	
        P_exp=[0.101325,5.90,7.04,8.36,10.38]
        
        # Condo et al.			
        Tg_exp_condo = [373.18,328.40,308.21,305.84,304.49]
        P_exp_condo = [0.100,6.072,7.139,8.419,10.440]
        
        # Pham el al. 90 nm				
        Tg_exp_pham90nm = [383.11,348.03,322.87,308.10,298.06]
        P_exp_pham90nm = [0.089,3.990,4.792,5.209,4.884]
       
        # Pham et al. 17 nm				
        Tg_exp_pham17nm = [362.47,347.58,322.65,307.98,297.61]
        P_exp_pham17nm = [0.089,1.278,3.218,4.071,3.675]
      
        # Wissinger Bulk				
        Tg_exp_wissinger = [373.86,338.67,324.00,308.89]	
        P_exp_wissinger = [0.100,3.645,4.864,6.072]	

        # Tg_exp_Condo=[374.1,339.2,323.9,308.8]
        # P_exp_Condo=[0.008250,3.64,4.88,6.06]

        #Experiment Data PS from Maria Pantoula's Paper Ref. Zhang[32]
        Tg_exp=	[375.498,372.188,367.621,360.073,354.183,346.24,337.037,333.065]
        P_exp = [0.362753,0.730141,1.4465,2.0206,3.11808,4.20183,4.8678,5.38213]

        #Experiment Data PS from Maria Pantoula's Paper Ref. Tsivintzelis[33]
        Tg_exp=	[348.288,339.218,318.033]
        P_exp = [2.99885,4.00457,6.00232]

        #Experiment Data PS from Maria Pantoula's Paper Ref. O'Neill[36]
        Tg_exp=	[372.121,367.42,362.852,354.444,344.78,330.614,321.212]
        P_exp = [0.541896,0.909318,1.36397,2.27326,3.59581,5.26281,6.07571]

        #Experiment Data PS from Maria Pantoula's Paper Ref. Wissinger[18]
        Tg_exp=	[377.218,338.289,323.326,308.297]
        P_exp = [0.0137675,3.65106,4.85895,6.0852]

    if Polymer_Type=='PMMA' and Solvent=='CO2':

        #Experiment Data PMMA from Condo Paper;
        Tg_exp=	[378.0,277.8,297.7,317.7,337.8,347.9]
        P_exp=	[0.101325,3.75,5.14,5.70,5.09,3.73]

        #Experiment Data PMMA from Ruosong et al.
        Tg_exp = [368.3,352.3,346.9,326.5,323.2,311.4,317.9,320.1,320.9,322.2,323.1,296.2,315.6]
        P_exp = [2,4,6,8,10,12,14,16,18,20,22,4,6]

        #Experiment Data PMMA from Maria Pantoula's Paper Ref. Kamiya [24]
        Tg_exp = [358.138,348.031,338.091,308.019]
        P_exp = [3.60067,3.90142,4.89557,5.00418]
        #Experiment Data PMMA from Maria Pantoula's Paper Ref. Wissinger[18]
        Tg_exp = [331.826,314.618,305.68]
        P_exp = [3.90142,3.96825,3.90142]
        #Experiment Data PMMA from Maria Pantoula's Paper Ref. Alessi[34]
        Tg_exp = [312.697,307.434]
        P_exp = [7.96992,10.0167]
        #Experiment Data PMMA from Maria Pantoula's Paper Ref. O'Neill[36]
        Tg_exp = [364.905,352.291,345.692,341.181,330.907]
        P_exp = [0.85213,1.77945,   2.23893,2.92398,3.67586]

        #Experiment Data PMMA from Chul Parks's Paper Ref. Condo
        Tg_exp = [367.2229,358.135,349.1906,346.4349,341.2781,333.2876,325.3884,313.574,306.2017,298.0507,290.2496,275.85918,271.3099]
        P_exp = [0,1.516703528,2.793439058,3.283467023,4.14107169,4.898557125,5.401372305,5.77479546,5.413126005,4.894818233,4.148741993,3.295585493,2.796499073]

        #Experiment Data PMMA from Chul Park's Paper Ref. Handa
        Tg_exp = [349.2809,338.7512,330.6587,319.2883,308.4899,298.0621,288.2614,278.34445]
        P_exp = [3.655390568,5.140430033,5.917521855,5.872158653,5.96386791,5.147036423,4.548094215,3.652857443]

        #Experiment Data PMMA from Dehua Liu's Paper Ref. His Work
        Tg_exp = [388.597,383.74,380.03,374.769,359.1871,344.4148]
        P_exp = [0.0988235,1.50706,2.24412,3.01,4.36882,5.99941]

        #Experiment Data PMMA from Dehua Liu's Paper Ref. Handa
        Tg_exp = [363.8414,354.1281,342.3911,336.9274,332.6778,324.1787]
        P_exp = [0.0947059,0.916177,1.88588,2.31824,3.04706,3.80059]

    if Polymer_Type=='PC' and Solvent=='CO2':

        #Experiment Data PC;
        Tg_exp = [423.02,413.201,401.982,394.405,383.186,417.975,414.327,407.875,399.741,396.934,368.8913,363.2814,348.1364,334.1149,268.80497,253.3957,242.7511]
        P_exp =	[0.0,1.506763545,2.82165807,4.375730258,5.586016853,0.0,0.699796046,1.641171158,2.642292555,3.24000873,5.675426033,6.377709608,8.006367128,9.231548498,9.992580308,8.976108173,7.98959784]

    return (Ppstar,Tpstar,Rpstar,Mp,Psstar,Tsstar,Rsstar,Ms,P_exp,Tg_exp)
