export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
#export CUDA_VISIBLE_DEVICES=1

#TEST='--skip-test'
#TEST='--only-test' 


#CONFIG_FILE='Stanford_walls/s_wall_Fpn432_bs1_lr20_CB.yaml'
CONFIG_FILE='Stanford_walls/s_3c_Fpn432_bs1_lr20_CB.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

