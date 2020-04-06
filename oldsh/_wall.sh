export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
export CUDA_VISIBLE_DEVICES=1

#TEST='--skip-test'
#TEST='--only-test' 


#CONFIG_FILE='walls/wall_Fpn4321_bs1_lr5.yaml'
#CONFIG_FILE='walls/wall_Fpn4321_bs1_lr5_corsem.yaml'
#CONFIG_FILE='walls/wall_Fpn4321_bs1_lr5_Rpn.yaml'
#CONFIG_FILE='walls/wall_Fpn4321_bs1_lr5_CB.yaml'

#CONFIG_FILE='walls/SD_wall_Fpn4321_bs1_lr20.yaml'
#CONFIG_FILE='walls/SD_wall_Fpn4321_bs1_lr20_corsem.yaml'
#CONFIG_FILE='walls/SD_wall_Fpn4321_bs1_lr20_Rpn.yaml'
CONFIG_FILE='walls/SD_wall_Fpn4321_bs1_lr20_CB.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

