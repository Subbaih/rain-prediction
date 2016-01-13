# Preprocess and Prepare train
echo 'Pre-processing for training'
date
# Preprocess and Prepare train
cd data/train
python ../../main.py create_chunks 'train.csv' 1;
nfeatures=22;
for i in `seq 1 $nfeatures`
	do 
		python ../../main.py process_chunks $i 1
	done
wait
date
cd ../..
echo 'Pre-processing for training done'

cd data/test
echo 'Pre-processing for testing'
python ../../main.py create_chunks 'test.csv' 0;
for i in `seq 1 $nfeatures`
	do 
		python ../../main.py process_chunks $i 0 
	done
wait
date
cd ../..
echo 'Pre-processing for testing done'


echo 'Feature generation for training'
# Feature generation and Merging
cd data/train
python ../../main.py gen_features 1 1;   # Creates exp_train.csv as well
python ../../main.py merge_features 'features_train.csv'  
echo 'Feature generation for training done'
cd ../..

echo 'Feature generation for testing'
# Feature generation and Merging
cd data/test
python ../../main.py gen_features 0 0;
python ../../main.py merge_features 'features_test.csv' 
echo 'Feature generation for testing done'
cd ../..

echo 'Cross validating'
# Cross Validate
#python main.py cross_validate "data/train/features_train.csv" "data/train/exp_train.csv" "data/cross_validation.result" 1 
#echo 'Cross validating done'

# Actual Run
echo 'Training'
python main.py train "data/train/features_train.csv" "data/train/exp_train.csv" "data/trained.model" 1 
echo 'Training done'
echo 'Predicting'
python main.py predict "data/trained.model" "data/test/features_test.csv" "data/output.csv" 1
echo 'Predicting done'

