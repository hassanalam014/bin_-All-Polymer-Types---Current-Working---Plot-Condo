# Author: Kier von Konigslow
# Date: November 2017
#

###########################################################################
# Program #1:
###########################################################################
import sys
import colorama

colorama.init()

def prRed(x):
	print('\033[91m {}\033[00m' .format(x))
def prGreen(x):
	print('\033[92m {}\033[00m' .format(x))
def prYellow(x):
	print('\033[93m {}\033[00m' .format(x))
def prLightPurple(x):
	print('\033[94m {}\033[00m' .format(x))
def prPurple(x):
	print('\033[95m {}\033[00m' .format(x))
def prCyan(x):
	print('\033[96m {}\033[00m' .format(x))
def prLightGray(x):
	print('\033[97m {}\033[00m' .format(x))
def prBlack(x):
	print('\033[98m {}\033[00m' .format(x))

# direction='bwd'
# Tg=100.0
# prRed('Hurry! Tg is:{} for direction {}'.format(Tg,direction))
# print 'Hurry! Tg is:', Tg, 'for direction', direction
# prCyan('Hurry')
# prYellow('Failed')
# prGreen('Hurry')
# prRed('Hurry')
# prGreen('Hurry')
'''
###########################################################################
# or Use:      Program #2:
###########################################################################
import sys
from termcolor import colored, cprint 
import colorama
colorama.init()

text = colored('Hello, World!', 'red', attrs=['reverse', 'blink']) 
print(text) 
cprint('Hello, World!', 'green', 'on_red') 
  
print_red_on_cyan = lambda x: cprint(x, 'red', 'on_cyan') 
print_red_on_cyan('Hello, World!') 
print_red_on_cyan('Hello, Universe!') 
  
for i in range(10): 
    cprint(i, 'magenta', end=' ') 
  
cprint("Attention!", 'red', attrs=['bold'], file=sys.stderr) 
'''