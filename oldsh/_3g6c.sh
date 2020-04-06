export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
export CUDA_VISIBLE_DEVICES=1
#export Application.verbose_crash=True

#TEST='--skip-test'
TEST='--only-test' 


#CONFIG_FILE='3g6cs/3g6c_Fpn4321_bs1_lr5.yaml'
#CONFIG_FILE='3g6cs/3g6c_Fpn4321_bs1_lr5_corsem.yaml'
#CONFIG_FILE='3g6cs/3g6c_Fpn4321_bs1_lr5_Rpn.yaml'
#CONFIG_FILE='3g6cs/3g6c_Fpn4321_bs1_lr5_CB.yaml'

#CONFIG_FILE='3g6cs/SD_3g6c_Fpn4321_bs1_lr20.yaml'
#CONFIG_FILE='3g6cs/SD_3g6c_Fpn4321_bs1_lr20_corsem.yaml'
#CONFIG_FILE='3g6cs/SD_3g6c_Fpn4321_bs1_lr20_Rpn.yaml'
#CONFIG_FILE='3g6cs/SD_3g6c_Fpn4321_bs1_lr20_CB.yaml'
#CONFIG_FILE='3g6cs/SD_3g6c_Fpn4321_bs1_lr5_corsem.yaml'

CONFIG_FILE='3g6cs/Ja_3g6c_Fpn4321_bs1_lr10_CB.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

