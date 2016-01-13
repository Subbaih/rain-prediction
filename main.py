import sys;
import pandas as pd;
import numpy as np;
#from sklearn import preprocessing as skpp;
#from sknn.mlp import Regressor, Layer;
from rain_preprocess import *;
from rain_train import *;
#from rain_predict import *;
from rain_feature import *;
#from rain_postprocess import *;

if __name__ == '__main__':
	globals()[sys.argv[1]](*sys.argv[2:]);
