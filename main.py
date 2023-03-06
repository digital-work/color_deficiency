'''
Created on 7. mars 2014

@author: Joschua
'''

import test_wech
import sys, getopt
from settings import test_functions

## This is the original main function.

def main(argv):
   s = 0
   inputfile = ""
   experiment_id = 0
   #method = "test0"
   dict = {}
   methodToCall = None
   help = False
   try:
      opts, args = getopt.getopt(argv,"hi:s:e:f:c:u:p:",["ifile=","size=","exp=","function=","coldef_types=","simulation_types=","path="])
   except getopt.GetoptError:
      print 'main.py -i <size>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
          help = True
      elif opt in ("-f", "--function"):
          # Calls the method from the module test_wech
          if arg.isdigit() and (int(arg) in test_functions):
              method = "test_wech"+arg
              methodToCall = getattr(test_wech, method)
          else:
              print arg
              print "Error: Wrong ID for test_wech function chosen. The ID for the test_wech function has to be an integer from following array: "+str(test_functions)
              sys.exit(2)
      elif opt in ("-s", "--size"):
          # max size of the image(s) being computed
          dict['size'] = int(arg)
      elif opt in ("-i", "--ifile"):
          # inputfile(s) for the test_wech methods
          dict['inputfiles'] = arg
      elif opt in ("-e", "--exp"):
          print arg
          # ID of the experiment being computed. O for all experiments
          dict["experiment_id"] = arg
      elif opt in ("-c", "--coldef_types"):
          # Color deficiency types used in the experiment
          dict["coldef_types"] = arg.split(',')
      elif opt in ("-u", "--simulation_types"):
          # Simulation types used in the experiment
          dict["simulation_types"] = arg.split(',')
      elif opt in ("-p", "--path"):
          # Simulation types used in the experiment
          dict["path"] = arg
   #print 'Size is ', s
   
   #test_wech.test8(inputfile,s)
   #test_wech.test14()
   #methodToCall = getattr(test_wech, method)
   #result = methodToCall(dict)
   #test_wech."test15"(experiment_id)
   
   if methodToCall:
       if help:
           print methodToCall.__doc__
           sys.exit(2)
       else:
           result = methodToCall(dict)
   else:
       print "Error: No method chosen. You did not specify any function to call. Please choose a function with \'-f\' foloowed by an integer from the following array: "+str(test_functions)

if __name__ == "__main__":
   main(sys.argv[1:])


