import pandas as pd;
import numpy as np;
import sys;
import re;
import os;
import random;
from sklearn import preprocessing as skpp;
from sklearn.ensemble import *;
#from sknn.mlp import Regressor, Layer;
import xgboost as xgb;
import pickle;
from sklearn.externals import joblib;
from sklearn.grid_search import *;
import rain_feature;

def cross_validate(Xtrain,Ytrain,op_file='run.cv',isfile=True):
	isfile = bool(int(isfile))
	if isfile:
		Xtrain = pd.read_csv(Xtrain);
		Ytrain = pd.read_csv(Ytrain);
	Xtrain = rain_feature.filter_features(Xtrain);
	Xtrain = Xtrain.set_index('Id');
	Xtrain[np.isnan(Xtrain)] = 9999;
	Xtrain[np.isinf(Xtrain)] = 9999;
	Ytrain = Ytrain.set_index('Id');
	Ytrain.fillna(9999,inplace=True);
	
	# Run SciKit Cross Validation for Gradient Boosting Regressor 
	GBR_cv(Xtrain,Ytrain,op_file);

	# Needed for XGBoost 
	'''
	num_over = None; num_threads  = 15; K=5;
	param = {};
	param['eta'] = 0.015; param['gamma']  = 1.5; param['max_depth'] = 9;
	param['min_child_weight'] = 55; param['subsample'] = 0.45; param['colsample'] = 0.55;
	param['nthread'] = num_threads; param['objective'] = 'reg:linear'; param['eval_metric'] = 'rmse';
	param['silent'] = 1; 
	history = cv_tree_xgb(Xtrain, Ytrain, param, num_over, K);
	print history;
	'''

def divide_for_train(X,Y):
	rows = random.sample(X.index,len(X.index)/2);
	Xtest = X.ix[rows];
	Xtrain = X.drop(rows);
	Ytest = Y.ix[rows];
	Ytrain = Y.drop(rows);
	return [Xtrain,Ytrain,Xtest,Ytest]; 

def train(Xtrain,Ytrain,fmodel,isfile=True):
	isfile = bool(int(isfile));
	if isfile:
		Xtrain = pd.read_csv(Xtrain);
		Ytrain = pd.read_csv(Ytrain);
	Xtrain = rain_feature.filter_features(Xtrain);
	Xtrain = Xtrain.set_index('Id'); 
	Xtrain[np.isnan(Xtrain)] = 9999;
	Xtrain[np.isinf(Xtrain)] = 9999;
	Ytrain = Ytrain.set_index('Id');
	Ytrain.fillna(9999,inplace=True);
	
	# Needed for SciKit Gradient Boosting Regressor
	print 'Read the input: %d'%len(Xtrain);
	learning_rate = 0.1; ntrees = 1000; num_features = 5; max_depth = 8; alpha = 0.9; v = 1;
	gbr = GBR_train(Xtrain, Ytrain, learning_rate, ntrees, num_features, max_depth, alpha);
	gbr_save_model(gbr,fmodel);

	# Needed for XGBoost 
	'''
	[Xtrain,Ytrain,Xtest,Ytest] = divide_for_train(Xtrain,Ytrain);
	num_over = 10; num_threads  = 10;
	param = {};
	param['eta'] = 0.015; param['gamma']  = 1.5; param['max_depth'] = 9;
	param['min_child_weight'] = 55; param['subsample'] = 0.45; param['colsample'] = 0.55;
	param['nthread'] = num_threads; param['objective'] = 'reg:linear'; param['eval_metric'] = 'rmse';
	param['silent'] = 1;
	XGTree = train_tree_xgb(Xtrain, Ytrain, Xtest, Ytest, param, num_over);
	XGTree.save_model(fmodel);
	'''

def xgboost_load_model(path):
	bst = xgb.Booster();
	bst.load_model(path);
	return bst;

def gbr_save_model(gbr,fpath):
	#pickle.dump(gbr,open(fpath,'w'));
	joblib.dump(gbr,fpath);

def gbr_load_model(fpath):
	gbr = joblib.load(fpath);
	#gbr = pickle.load(open(fpath,'r'));
	return gbr;

def predict(fmodel,testX,op_file='data/output.csv',isfile=True):
	if isfile:
		testX = pd.read_csv(testX);
	testX = rain_feature.filter_features(testX);
	testX = testX.set_index('Id');
	testX[np.isnan(testX)] = 9999;
	testX[np.isinf(testX)] = 9999;
	print "Predicting";

	'''
	bst = xgboost_load_model(fmodel);
	xg_val = xgb.DMatrix(testX.as_matrix(),missing=np.nan);
	pred = bst.predict(xg_val);
	'''

	gbr = gbr_load_model(fmodel);
	pred = GBR_predict(gbr,testX);

	pred = pd.DataFrame(pred);
	pred.index = range(1,len(pred)+1);
	pred.columns = ['Expected'];
	pred.index.name = 'Id';
	pred.to_csv(op_file);

def mean_abs_err(Yhat,X):
	Y = X.get_label();
	mae = np.mean(abs(Y-Yhat));
	return 'MAE',mae;

def cv_tree_xgb(Xtrain, Ytrain, param, num_over, K, num_round = 10000):
	eval_func = mean_abs_err;
	XGtrain = xgb.DMatrix(Xtrain.as_matrix(), label=Ytrain.as_matrix(), missing=np.nan);
	print mean_abs_err(XGtain.get_label(),XGtrain);
	#print XGtrain.get_labels();
	return xgb.cv(param, XGtrain, num_round, nfold=K, feval=eval_func, early_stopping_rounds=num_over, show_progress=True, show_stdv=True);

def train_tree_xgb(Xtrain, Ytrain, Xtest, Ytest, param, num_over, num_round = 10000):
	eval_func = mean_abs_err;
	print "Entered for training"
	XGtrain = xgb.DMatrix(Xtrain.as_matrix(), label=Ytrain.as_matrix(), missing=np.nan);
	XGtest = xgb.DMatrix(Xtest.as_matrix(), label=Ytest.as_matrix(), missing=np.nan);
	watchlist = [(XGtrain, 'train'),(XGtest, 'test')];
	#print('will wait {0} has no {1}\n'.format(watchlist[-1][1]))
	bst = xgb.train(param, XGtrain, num_round, watchlist, feval=eval_func, early_stopping_rounds=num_over,verbose_eval=True);
	return bst;


def GBR_train(Xtrain, Ytrain, lrate, num_trees, num_features, max_tree_depth, qalpha, sub_sample=0.8, v=1):
	gbr = GradientBoostingRegressor(loss='quantile', learning_rate=lrate, n_estimators=num_trees, max_features=num_features, max_depth=max_tree_depth, alpha=qalpha, subsample=sub_sample, verbose=v);
	gbr.fit(Xtrain,Ytrain);	
	return gbr;

def GBR_predict(gbr,Xtest):
	return gbr.predict(Xtest);

def GBR_cv(Xtrain,Ytrain,op_file,nthreads=5,v=1,K=5):
	param_grid = { 'learning_rate': [0.1,0.01,0.001],
		       'n_estimators': [500,1000,2000],
		       'max_depth': [4,8,12],
		       'min_samples_leaf': [50,100,200],
		       'max_features':[5,10,15,25],
		     };
	Xtrain = Xtrain.as_matrix(); Ytrain = Ytrain.as_matrix();
	gbr = GradientBoostingRegressor(loss='quantile');
	gs_cv = GridSearchCV(gbr,param_grid,scoring=None,pre_dispatch=nthreads,verbose=v,cv=K).fit(Xtrain,Ytrain);
	print gs_cv.best_params;
	joblib.dump(gs_cv,op_file);
