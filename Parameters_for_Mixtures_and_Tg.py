#############################################################################################
# Polymer_Type='PMMA' #PS or PMMA or DME or LPP or BPP or PLA or LDPE
# Solvent='CO2' 
#############################################################################################

def Parameters_for_Mixtures_and_Tg(**kwargs):
	
    for key,value in kwargs.items():
		exec "%s='%s'" % (key,value)
    
    print 'Polymer Type', Polymer_Type, 'Solvent', Solvent
    print 'PVT and Tg Data from', Parameters_Paper, 'Cp Data from', Cp_Polymer_Weight

    zeta=0.0            #Incase it is not written
    delta=0.0            #Incase it is not written
    
    if Kier or Hassan or Hassan_Var_Vol:

        if Polymer_Type == 'PLA' and  Parameters_Paper == 'Self_Park':
            g = 0.9603539685008077          #Random Values
            epsilon_p = 8184.287109670879   #Random Values
            x = 0.3094949494949495          #Random Values

        if Polymer_Type == 'PC' and  Parameters_Paper == 'Self_Aravind' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 0.9603539685008077
            epsilon_p = 8184.287109670879
            x = 0.3094949494949495

        if Polymer_Type == 'PC' and  Parameters_Paper == 'Self_Aravind' and  Cp_Polymer_Weight == '03kilo_POST_THESIS':
            g = 0.886828162139225
            epsilon_p = 8094.950628062509
            x = 0.3071717171717172

        if Polymer_Type == 'PC' and  Parameters_Paper == 'Self_Sato' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 0.9160630513838226
            epsilon_p = 8344.843345144749
            x = 0.3118181818181818

        if Polymer_Type == 'PC' and  Parameters_Paper == 'Self_Sato' and  Cp_Polymer_Weight == '03kilo_POST_THESIS':
            g = 0.8463051587118186
            epsilon_p = 8260.004839235002
            x = 0.3094949494949495

        if Polymer_Type == 'PC' and  Parameters_Paper == 'Self_Rudolph' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 1.0778426983109262
            epsilon_p = 8269.988678082645
            x = 0.35363636363636364

        if Polymer_Type == 'PC' and  Parameters_Paper == 'Self_Rudolph' and  Cp_Polymer_Weight == '03kilo_POST_THESIS':
            g = 0.9941838957452115
            epsilon_p = 8133.359983178601
            x = 0.35363636363636364

        if Polymer_Type == 'PC' and  Parameters_Paper == 'Self_Kikuchi' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 0.3846782523712565
            epsilon_p = 7598.318940401397
            x = 0.3251515151515152

        if Polymer_Type == 'PC' and  Parameters_Paper == 'Self_Kikuchi' and  Cp_Polymer_Weight == '03kilo_POST_THESIS':
            g = 0.3576091464401348
            epsilon_p = 7580.047801610359
            x = 0.32343434343434346

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Schmidt' and  Cp_Polymer_Weight == '01kilo_POST_THESIS':
            g = 1.1803729078513423
            epsilon_p = 7736.482977361391
            x = 0.31646464646464645

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Schmidt' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 1.0050426359975553
            epsilon_p = 7561.237283183051
            x = 0.3118181818181818

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Schmidt' and  Cp_Polymer_Weight == '03kilo_POST_THESIS':
            g = 1.005255413264288
            epsilon_p = 7561.63751151864
            x = 0.3118181818181818

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Schmidt' and  Cp_Polymer_Weight == '04kilo_POST_THESIS':
            g = 1.9173410467897665
            epsilon_p = 8287.574183619457
            x = 0.33272727272727276

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Grassia' and  Cp_Polymer_Weight == '01kilo_POST_THESIS':
            g = 1.1778296636402585
            epsilon_p = 7167.713238230759
            x = 0.3094949494949495

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Grassia' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 1.0029675775730258
            epsilon_p = 7054.874048684184
            x = 0.30252525252525253

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Grassia' and  Cp_Polymer_Weight == '03kilo_POST_THESIS':
            g = 1.0031801233904134
            epsilon_p = 7055.260307170906
            x = 0.30252525252525253

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Grassia' and  Cp_Polymer_Weight == '04kilo_POST_THESIS':
            g = 1.91293927392975
            epsilon_p = 7709.391692869333
            x = 0.3257575757575758

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Walsh' and  Cp_Polymer_Weight == '01kilo_POST_THESIS':
            g = 1.204627147640183
            epsilon_p = 7691.665218192569
            x = 0.3211111111111111

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Walsh' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 1.0252858795727555
            epsilon_p = 7515.57495475635
            x = 0.31646464646464645

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Walsh' and  Cp_Polymer_Weight == '03kilo_POST_THESIS':
            g = 1.0255030120716329
            epsilon_p = 7515.972448561153
            x = 0.31646464646464645

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Walsh' and  Cp_Polymer_Weight == '04kilo_POST_THESIS':
            g = 1.959779023769397
            epsilon_p = 8300.624250858695
            x = 0.33505050505050504

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Wen' and  Cp_Polymer_Weight == '01kilo_POST_THESIS':
            g = 1.5947153846284419
            epsilon_p = 8164.296491860293
            x = 0.3187878787878788

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Wen' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 1.3485687048430275
            epsilon_p = 7939.058532211139
            x = 0.31414141414141417

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Wen' and  Cp_Polymer_Weight == '03kilo_POST_THESIS':
            g = 1.3488656087855613
            epsilon_p = 7939.507018517817
            x = 0.31414141414141417

        if Polymer_Type == 'PMMA' and  Parameters_Paper == 'Self_Wen' and  Cp_Polymer_Weight == '04kilo_POST_THESIS':
            g = 2.6543781198849605
            epsilon_p = 8872.754945514838
            x = 0.33505050505050504

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Kier' and  Cp_Polymer_Weight == '00kilo_POST_THESIS':
            g = 1.3412992033448965
            epsilon_p = 7879.151565939133
            x = 0.3187878787878788

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Kier' and  Cp_Polymer_Weight == '01kilo_POST_THESIS':
            g = 1.1956168506002003
            epsilon_p = 7776.119618226106
            x = 0.31414141414141417

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Kier' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 1.1784266669546997
            epsilon_p = 7747.72864311371
            x = 0.31414141414141417

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Kikuchi' and  Cp_Polymer_Weight == '00kilo_POST_THESIS':
            g = 0.771522067347692
            epsilon_p = 7314.702112102635
            x = 0.3280808080808081

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Kikuchi' and  Cp_Polymer_Weight == '01kilo_POST_THESIS':
            g = 0.6929792369704724
            epsilon_p = 7294.718193970311
            x = 0.32343434343434346

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Kikuchi' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 0.6836410840911611
            epsilon_p = 7272.774565922051
            x = 0.32343434343434346

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Park' and  Cp_Polymer_Weight == '00kilo_POST_THESIS':
            g = 2.9261596544846458
            epsilon_p = 8378.749749148625
            x = 0.3094949494949495

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Park' and  Cp_Polymer_Weight == '01kilo_POST_THESIS':
            g = 2.5693905049844674
            epsilon_p = 8180.677416577772
            x = 0.30484848484848487

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Park' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 2.5278250145951935
            epsilon_p = 8144.32790306501
            x = 0.30484848484848487

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Grassia' and  Cp_Polymer_Weight == '00kilo_POST_THESIS':
            g = 1.1645777519418357
            epsilon_p = 7197.126197552568
            x = 0.31414141414141417

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Grassia' and  Cp_Polymer_Weight == '01kilo_POST_THESIS':
            g = 1.0403241128054486
            epsilon_p = 7112.391920660067
            x = 0.3094949494949495

        if Polymer_Type == 'PS' and  Parameters_Paper == 'Self_Grassia' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            g = 1.0256557135378013
            epsilon_p = 7086.5616869828045
            x = 0.3094949494949495

        # Binary Paramters:
        if Polymer_Type=='PMMA' and Solvent=='CO2' and Parameters_Paper=='Self_Grassia' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            if Paper_Number == 'Paper15':
                zeta = 0.94393585   #+/- 0.01691760 (1.79%) (init = 1)
                delta = 0.88621038  #+/- 0.03257666 (3.68%) (init = 1)
                ##### Below is for delta = 1.0
                # zeta = 1.00191274 #+/- 0.00138619 (0.14%) (init = 0.93)
                # delta = 1.0       #(fixed)
                ##### Below is for vhm in between vs and vp:
                # zeta = 1.00191251 #+/- 0.03942875 (3.94%) (init = 1.1)
                # delta = 1.00000000 #+/- 0.07766170 (7.77%) (init = 1.344)
        if Polymer_Type=='PS' and Solvent=='CO2' and Parameters_Paper=='Self_Grassia' and  Cp_Polymer_Weight == '02kilo_POST_THESIS':
            if Paper_Number == 'Paper4_11_12':
                zeta = 1.07136875   #+/- 0.04821701 (4.50%) (init = 1)
                delta = 1.23734185  #+/- 0.10479860 (8.47%) (init = 1)
                ##### Below is for delta = 1.0
                # zeta = 0.96242124 #+/- 0.00242216 (0.25%) (init = 1.07)
                # delta = 1.0      #(fixed)
                ##### Below is for vhm in between vs and vp:
                # zeta = 1.07136995 #+/- 0.04821750 (4.50%) (init = 0.95)
                # delta = 1.23734449 #+/- 0.10480090 (8.47%) (init = 1.344)

            if Paper_Number == 'Paper11_13':
                zeta = 0.95125453   #+/- 0.03123427 (3.28%) (init = 1)
                delta = 0.96557611  #+/- 0.06870099 (7.12%) (init = 1)
            if Paper_Number == 'Paper11_13_LowTemp':
                zeta = 0.95125453   #+/- 0.03123427 (3.28%) (init = 1)
                delta = 0.96557611  #+/- 0.06870099 (7.12%) (init = 1)    
            if Paper_Number == 'Paper11_12':
                zeta = 1.09730204   #+/- 0.07209759 (6.57%) (init = 1)
                delta = 1.30622455  #+/- 0.15997201 (12.25%) (init = 1)
            if Paper_Number == 'Paper12_13':
                zeta = 1.04532004   #+/- 0.03281026 (3.14%) (init = 1)
                delta = 1.19773798  #+/- 0.07219192 (6.03%) (init = 1)


        ###################################################
        #CONDO BELOW:
        if Polymer_Type=='PMMA' and Solvent=='CO2' and Parameters_Paper=='Self_Grassia':
            if Paper_Number == 'Paper15':       #Solubility Data Paper to Regress zeta
                czeta = 1.02734491 #+/- 5.5474e-04 (0.05%) (init = 1)
        if Polymer_Type=='PMMA' and Parameters_Paper=='Self_Grassia':
            cepsilon_s = 0.0
            cepsilon_p = 4920.0
            cz = 4.00

        if Polymer_Type=='PS' and Solvent=='CO2' and Parameters_Paper=='Self_Grassia':
            if Paper_Number == 'Paper4_11_12':       #Solubility Data Paper to Regress zeta
                czeta = 1.00369516 #+/- 7.6674e-04 (0.08%) (init = 1)
        if Polymer_Type=='PS' and Parameters_Paper=='Self_Grassia':
            cepsilon_s = 0.0
            cepsilon_p = 5028.0
            cz = 4.00

        ####################################################################################################################
        # All Pre-Thesis Starts From Below:
        ####################################################################################################################
    if False: #Making Pre-Thesis False
        if Polymer_Type=='PS' and Cp_Polymer_Weight=='Pre-Thesis':
            g=1.67 
            epsilon_p=8013 
            x=0.311

        if Polymer_Type=='PMMA' and Cp_Polymer_Weight=='Pre-Thesis':	
            g=1.66 
            epsilon_p=8094 
            x=0.323

        if Polymer_Type=='PC' and Cp_Polymer_Weight=='Pre-Thesis':
            g=0.84 
            epsilon_p=8273.0
            x=0.317

        if Condo_Original or Condo:

            if Polymer_Type=='PLA':
                cepsilon_s=0            #Random Values
                cepsilon_p=7151.0       #Random Values
                cz=5.0                  #Random Values

            if Polymer_Type=='PS':
                cepsilon_s=0
                cepsilon_p=7151.0
                cz=5.0

            if Polymer_Type=='PMMA':
                cepsilon_s=0
                cepsilon_p=7443.0
                cz=5.0

            if Polymer_Type=='PC':
                cepsilon_s=0
                cepsilon_p=6247.0
                cz=4.0

        if Kier and Cp_Polymer_Weight=='Pre-Thesis':

            if Polymer_Type=='PS' and Solvent=='CO2' and Parameters_Paper=='Kier':
                zeta=1.02127006
                delta=0.88102354

            if Polymer_Type=='PS' and Solvent=='CO2' and Parameters_Paper=='Condo':
                zeta=1.08820786     #1.08820786(Original)   #1.11000815(from Xs condo formula)
                delta=0.97423316    #0.97423316(Original)   #1.00003292(from Xs condo formula)

            if Polymer_Type=='PMMA' and Solvent=='CO2' and Parameters_Paper=='Condo':
                zeta=1.13500002       #1.135      #1.13500002(from Xs condo formula)		#1.12272978	#1.07643522	#	#1.1684#1.135			#1.10752004		#1.08621732		#1.135		
                delta=1.00003613      #1.00       #1.00003613(from Xs condo formula)		#0.73223348	#1.0		#	#0.5#0.74264552			#0.90758208		#1.05473203		#1.00	

            if Polymer_Type=='PC' and Solvent=='CO2':
                zeta=1.06719283	#From Kier Solubility alpha
                # zeta=1.10694978 #From Condo Solubility alpha	
                delta=1.000

                #SuperCritical Solubility Fit by Condo Solubility alpha
                zeta=1.12596087
                delta=0.99999371

        if Condo or Condo_Original:

            if Polymer_Type=='PLA' and Solvent=='CO2':
                #Condo Zeta:
                czeta=1.1240			#Random Value

            if Polymer_Type=='PS' and Solvent=='CO2':
                #Condo Zeta:
                czeta=1.1240			#Condo Values. Correlation value: 1.110, iteration value: 1.1240

            if Polymer_Type=='PMMA' and Solvent=='CO2':
                #Condo Zeta:
                czeta=1.1350

            if Polymer_Type=='PC' and Solvent=='CO2':
                #Condo Zeta:
                czeta=1.000

    return cepsilon_s,cepsilon_p,cz,czeta,epsilon_p,g,x,delta,zeta