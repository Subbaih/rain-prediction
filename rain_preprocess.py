import sys;
import pandas as pd;
import numpy as np;
import os;
#from sklearn import preprocessing as skpp;
#from sknn.mlp import Regressor, Layer;

features = {
	0 : "Id",
	1 : "minutes_past",
	2 : "radardist_km",
	3 : "Ref",
	4 : "Ref_5x5_10th",
	5 : "Ref_5x5_50th",
	6 : "Ref_5x5_90th",
	7 : "RefComposite",
	8 : "RefComposite_5x5_10th",
	9 : "RefComposite_5x5_50th",
	10 : "RefComposite_5x5_90th",
	11 : "RhoHV",
	12 : "RhoHV_5x5_10th",
	13 : "RhoHV_5x5_50th",
	14 : "RhoHV_5x5_90th",
	15 : "Zdr",
	16 : "Zdr_5x5_10th",
	17 : "Zdr_5x5_50th",
	18 : "Zdr_5x5_90th",
	19 : "Kdp",
	20 : "Kdp_5x5_10th",
	21 : "Kdp_5x5_50th",
	22 : "Kdp_5x5_90th",
	23 : "Expected"
};

def clean_data(df):
	df = df[np.isfinite(df['Ref'])];
	return df;

def pre_process(ip_fname):
	X = pd.read_csv(ip_fname);
	X = clean_data(X);
	return X;

def create_partitions(ip_fname,col,op_fname,istrain,op_mode='a'):
	oX = pd.DataFrame(None);
	X = pd.read_csv(ip_fname);
	X = clean_data(X);
	# Create partitions - id,col,expected 
	if istrain:
		pX = X.loc[:,['Id',features[col],'Expected']];
	else:
		pX = X.loc[:,['Id',features[col]]];
	oX = pd.concat([oX,pX]);
	oX.to_csv(op_fname,index=False,mode=op_mode,header=False); # Different entries Written at diff times
	print 'Created %s'%(op_fname);

def create_chunks(ip_fname,istrain=True,num_grps=100000):
	num_grps = int(num_grps);
	istrain = bool(int(istrain));
	if istrain:
		prefix='chunks/train_'
	else:
		prefix='chunks/test_'
	testX = pd.read_csv(ip_fname);
	gtestX = testX.groupby('Id');
	count = 0; 
	newX = pd.DataFrame(None);
	for k in gtestX.groups.keys():
		rows = gtestX.groups[k];
		if len(newX) >= num_grps:
			count = count + 1;
			fname = prefix + str(count) + '.csv';
			print 'Writing %s'%(fname);
			newX.to_csv(fname,index=False);
			newX = pd.DataFrame(None);
		x = testX.loc[rows,:];
		newX = newX.append(x);
	if len(newX)!=0:
		count = count + 1;
		fname = prefix + str(count) + '.csv';
		print 'Writing %s'%(fname);
		newX.to_csv(fname,index=False);
	
def process_chunk(ip_fname,col,istrain):
	op_fname = 'partitions/'+('%s.csv'%(features[col]))
	print 'Output to %s processing from %s'%(op_fname,ip_fname);
	create_partitions(ip_fname,col,op_fname,istrain);

def process_chunks(col,istrain=True):
	col = int(col); istrain = bool(int(istrain));
	files = [f for f in os.listdir('chunks')];
	for i in range(0,len(files)):
		ip_fname = 'chunks/'+files[i];
		process_chunk(ip_fname,col,istrain);
