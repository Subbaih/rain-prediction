import os;
import sys;
import pandas as pd;
import numpy as np;
import rain_preprocess; 

def valid_time(mp):
	vt = np.zeros_like(mp);
	vt[0] = mp.iloc[0];
	for n in range(1,len(mp)):
		vt[n] = mp.iloc[n] - mp.iloc[n-1];
	vt[-1] = vt[-1] + 60 - np.sum(vt);
	vt = vt / 60.0;
	return vt;

def rate_kdp(x):
	szh = 0;
	col_mp = rain_preprocess.features[1]; col_kdp = rain_preprocess.features[19]; 
	x = x.sort(col_mp,ascending=True);
	mp = x[col_mp];
	kdp = x[col_kdp];
	vt = valid_time(mp);

	kdp_aa = 40.6; kdp_bb = 0.866; 
	s = 0;
	for xkdp, hrs in zip(kdp,vt):
		if np.isfinite(xkdp):
			mmperhr = np.sign(xkdp) * kdp_aa * (abs(xkdp) ** kdp_bb);
			s =  s + mmperhr * hrs;
	return s;

def rate_kdp_zdr(x):
	szh = 0;
	col_mp = rain_preprocess.features[1]; col_kdp = rain_preprocess.features[19]; col_zdr = rain_preprocess.features[15];
	x = x.sort(col_mp,ascending=True);
	mp = x[col_mp];
	kdp = x[col_kdp];
	zdr = x[col_zdr];
	vt = valid_time(mp);

	kdpzdr_aa = 136; kdpzdr_bb = 0.968; kdpzdr_cc = -2.86;
	s = 0;
	for xkdp, xzdr, hrs in zip(kdp,zdr,vt):
		if np.isfinite(xkdp) and np.isfinite(xzdr):
			mmperhr = np.sign(xkdp) * kdpzdr_aa * (abs(xkdp) ** kdpzdr_bb) * (xzdr ** kdpzdr_cc);
			s =  s + mmperhr * hrs;
	return s;

def rate_z_zdr(x):
	col_mp = rain_preprocess.features[1]; col_zh = rain_preprocess.features[3]; col_zdr = rain_preprocess.features[15];
	x = x.sort(col_mp,ascending=True);
	mp = x[col_mp];
	zh = x[col_zh];
	zdr = x[col_zdr];
	vt = valid_time(mp);

	zzdr_aa = 0.00746; zzdr_bb = 0.945; zzdr_cc = -4.76;
	s = 0;
	for xzh, xzdr, hrs in zip(zh,zdr,vt):
		if np.isfinite(xzh) and np.isfinite(xzdr):
			mmperhr = zzdr_aa * (xzh ** zzdr_bb) * (xzdr ** zzdr_cc);
			s =  s + mmperhr * hrs;
	return s;

def rate_zh(x):
	szh = 0;
	col_mp = rain_preprocess.features[1]; col_zh = rain_preprocess.features[3];
	x = x.sort(col_mp,ascending=True);
	mp = x[col_mp];
	zh = x[col_zh];
	vt = valid_time(mp);

	zh_aa = 0.027366; zh_bb = 0.69444;
	s = 0;
	for xzh, hrs in zip(zh,vt):
		if np.isfinite(xzh):
			mmperhr = zh_aa * (xzh ** zh_bb);
			s =  s + mmperhr * hrs;
	return s;

def feature_per_column_domain(istrain=True):
	mp = pd.read_csv('partitions/minutes_past.csv',header=None);
	kdp = pd.read_csv('partitions/Kdp.csv',header=None);
	zdr = pd.read_csv('partitions/Zdr.csv',header=None);
	ref = pd.read_csv('partitions/Ref.csv',header=None);

	if istrain:
		mp = mp.drop(mp.columns[2],1);
		kdp = kdp.drop(kdp.columns[2],1);
		zdr = zdr.drop(zdr.columns[2],1);
		ref = ref.drop(ref.columns[2],1);

	# Rate KDP
	X = pd.DataFrame(columns=['Id','minutes_past','Kdp']);
	X['minutes_past'] = mp[mp.columns[1]];
	X['Id'] = kdp[kdp.columns[0]];
	X['Kdp'] = kdp[kdp.columns[1]];
	gX = X.groupby('Id');
	nX = pd.DataFrame(gX.apply(rate_kdp));
	nX.index.name = 'Id';
	nX.columns = ['Rate_KDP'];
	nX.to_csv('Rate_KDP.csv');

	# Rate KDP_ZDR
	X = pd.DataFrame(columns=['Id','minutes_past','Kdp','Zdr']);
	X['minutes_past'] = mp[mp.columns[1]];
	X['Id'] = kdp[kdp.columns[0]];
	X['Kdp'] = kdp[kdp.columns[1]];
	X['Zdr'] = zdr[zdr.columns[1]];
	gX = X.groupby('Id');
	nX = pd.DataFrame(gX.apply(rate_kdp_zdr));
	nX.index.name = 'Id';
	nX.columns = ['Rate_KDP_ZDR'];
	nX.to_csv('Rate_KDP_ZDR.csv');

	# Rate Ref_ZDR
	X = pd.DataFrame(columns=['Id','minutes_past','Ref','Zdr']);
	X['minutes_past'] = mp[mp.columns[1]];
	X['Id'] = ref[ref.columns[0]]
	X['Ref'] = ref[ref.columns[1]];
	X['Zdr'] = zdr[zdr.columns[1]];
	gX = X.groupby('Id');
	nX = pd.DataFrame(gX.apply(rate_z_zdr));
	nX.index.name = 'Id';
	nX.columns = ['Rate_Z_ZDR'];
	nX.to_csv('Rate_Z_ZDR.csv');

	# Rate Ref
	X = pd.DataFrame(columns=['Id','minutes_past','Ref']);
	X['minutes_past'] = mp[mp.columns[1]];
	X['Id'] = ref[ref.columns[0]]
	X['Ref'] = ref[ref.columns[1]];
	gX = X.groupby('Id');
	nX = pd.DataFrame(gX.apply(rate_zh));
	nX.index.name = 'Id';
	nX.columns = ['Rate_ZH'];
	nX.to_csv('Rate_ZH.csv');

def feature_per_column_stats(fname,istrain=True,genY=False):
	istrain = bool(int(istrain));
	full_fname = 'partitions/' + fname;
	col = pd.read_csv(full_fname,header=None);
	#print full_fname;
	# comes without header
	if istrain:
		if genY:
			Y = pd.DataFrame(columns=['Id','Expected'])
			Y['Id'] = col[col.columns[0]];
			Y['Expected'] = col[col.columns[2]];
			gY = Y.groupby('Id').median();
			gY.to_csv('exp_train.csv');
		col = col.drop(col.columns[2],1) # Drop Expected Rainfall
			
	colname = os.path.splitext(fname)[0];
	gcol = col.groupby(col.columns[0]); # Group by Id
	ncol = gcol.sum();
	ncol.columns = ['sum_'+colname];
	ncol['mean_'+colname] = gcol.mean();
	ncol['median_'+colname] = gcol.median();
	#ncol['min_'+colname] = gcol.min();
	#ncol['max_'+colname] = gcol.max();
	ncol['std_'+colname] = gcol.std();
	#ncol['mad_'+colname] = gcol.mad();
	#ncol['skw_'+colname] = gcol.skw();
	ncol.index.name = 'Id';
	ncol.to_csv('expanded_'+fname);
	print 'Created %s'%('expanded_'+fname);

def gen_features(istrain=True,genY=False):
	istrain = bool(int(istrain));
	genY = bool(int(genY));
	for c in range(1,23):
		col = rain_preprocess.features[c];
		feature_per_column_stats(col+'.csv',istrain,genY);
		genY = False # Generate only once
	feature_per_column_domain(istrain);

def merge_features(op_fname):
	dfs = pd.DataFrame(columns=['Id']);
	for k in rain_preprocess.features.keys():
		if k not in [0,23]:
			col = rain_preprocess.features[k];
			fname = 'expanded_'+col+'.csv';
			print 'Processing: %s'%(fname);
			df = pd.read_csv(fname);
			dfs = pd.merge(dfs,df,on='Id',how='outer',suffixes=['','']);
	domain_dfs = pd.DataFrame(columns=['Id','Rate_KDP','Rate_KDP_ZDR','Rate_Z_ZDR','Rate_ZH']);
	nX = pd.read_csv('Rate_KDP.csv');
	domain_dfs['Id'] = nX['Id'];
	domain_dfs['Rate_KDP'] = nX['Rate_KDP'];
	nX = pd.read_csv('Rate_KDP_ZDR.csv');
	domain_dfs['Rate_KDP_ZDR'] = nX['Rate_KDP_ZDR'];
	nX = pd.read_csv('Rate_Z_ZDR.csv');
	domain_dfs['Rate_Z_ZDR'] = nX['Rate_Z_ZDR'];
	nX = pd.read_csv('Rate_ZH.csv');
	domain_dfs['Rate_ZH'] = nX['Rate_ZH'];
	dfs = pd.merge(dfs,domain_dfs,on='Id',how='outer',suffixes=['','']);
	dfs.set_index('Id');
	dfs.index.name = 'Id';
	dfs.to_csv(op_fname,index=False);

def filter_features(X):
	cols = [
		'sum_minutes_past','mean_minutes_past','median_minutes_past','std_minutes_past',
		'sum_Ref_5x5_10th','mean_Ref_5x5_10th','median_Ref_5x5_10th','std_Ref_5x5_10th',
		'sum_RefComposite_5x5_10th','mean_RefComposite_5x5_10th','median_RefComposite_5x5_10th','std_RefComposite_5x5_10th',
		'sum_RhoHV_5x5_10th','mean_RhoHV_5x5_10th','median_RhoHV_5x5_10th','std_RhoHV_5x5_10th',
		'sum_Zdr_5x5_10th','mean_Zdr_5x5_10th','median_Zdr_5x5_10th','std_Zdr_5x5_10th',
		'sum_Kdp_5x5_10th','mean_Kdp_5x5_10th','median_Kdp_5x5_10th','std_Kdp_5x5_10th'
		];
	for col in cols:
		X = X.drop(col,1);
	return X;	
