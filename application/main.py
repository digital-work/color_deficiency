'''
Created on 9. des. 2014

@author: joschua
'''

import getopt
import os, sys

from papers import EIXX2015_SChaRDa, EIXX2015_SaMSEM_ViSDEM
def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hv:p:",["version=", "project="])
    except getopt.GetoptError:
        print 'Error: Wrong syntax'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-v", "--version"):
            # 1. Get current version number
            dict['version_id'] = str(arg)
        elif opt in ("p", "--project"):
            # 2. Get project ID
            dict['project_id'] = str(arg)
        
        if dict.has_key('version_id'):
            version_id = dict['version_id']
        else:
            print "Error: No version ID chosen. main.py -v <version_id> -p <project_id>"
            sys.exit(2)
            
        if dict.has_key('project_id'):
            project_id = dict['project_id']
        else:
            print "Error: No project ID chosen. main.py -v <version_id> -p <project_id>"
            sys.exit(2)    
                
        # 3. Check out color_deficiency version
        os.system('git checkout '+dict['version'])       
        
        # 4. Call function
        if dict['project_id'] == 'eixx2015_samsem_visdem':
            EIXX2015_SaMSEM_ViSDEM()
        elif dict['project_id'] == 'eixx2015_scharda':
            EIXX2015_SChaRDa()
        

if __name__ == "__main__":
    main(sys.argv[1:])