import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# instalacion de camb 
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower
# print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))


