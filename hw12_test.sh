data_directory=$1
prediction_file=$2
model_name=raw_adap.pth

python3 hw12_test.py --model_checkpoint $model_name --dataroot $data_directory --output_csv $prediction_file
