import  glob, os
from shutil import copyfile

pathes = ['./RES/res_Sw4c_fpn432_bs1_lr1_T635']
pathes = glob.glob('./RES/res*')

for path in pathes:
  if os.path.exists(f'{path}/log.txt'):
    copyfile(f'{path}/log.txt', f'{path}/_log.txt')
  f = open(f'{path}/last_checkpoint', 'r')
  checkpoint = './'+f.readlines()[0]
  fnames = glob.glob(f'{path}/model_*.pth')
  final = f'{path}/model_final.pth'
  min_loss = f'{path}/model_min_loss.pth'
  for s in fnames:
    if s == checkpoint or s==final or s == min_loss:
      continue
    os.remove(s)
    print(f'{s} removed')
  print('clean ok')
