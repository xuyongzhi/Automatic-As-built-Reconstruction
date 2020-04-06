export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
#export CUDA_VISIBLE_DEVICES=1

TEST='--skip-test'
#TEST='--only-test' 

#CONFIG_FILE='1S_4c_fpn432_bs1.yaml'
CONFIG_FILE='1S_Sw4c_fpn432_bs1.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

