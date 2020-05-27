data_directory=$1
prediction_file=$2

python3 hw12_test.py --dataroot $data_directory --output_csv $prediction_file --use_gpu\
  --model_checkpoint resnet_18_adap_2500.pth resnet_34_adap_2000.pth resnet_18_adap_2000.pth\
  --resnet_type 18 34 18
