# xyz May 2019
import os
from scene_samples import SceneSamples
from shutil import copytree, rmtree

def read_obj(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    n = len(lines)

    #del_blocks = []
    find_end = 0
    lines_noceiling = []
    lines_ceiling = []
    for i in range(n):
        if find_end == 0 and  lines[i][0:9] == 'g Ceiling':
            del_start = i
            find_end = 1
        if find_end==1 and lines[i][0] !='g':
            find_end = 2
        if find_end == 2 and lines[i][0] =='g':
            del_end = i-1
            find_end = 0
            #del_blocks.append( (del_start, del_end) )

        if find_end == 2 and lines[i][0] == 'f':
            pass
        else:
            lines_noceiling.append(lines[i])

        if lines[i][0] == 'v' or  find_end >= 1:
            lines_ceiling.append(lines[i])

    #print(del_blocks)

    n_new = len(lines_noceiling)
    print(f'{n} -> {n_new}')

    new_fn = fn.replace('house.obj', 'no_ceiling_house.obj')
    with open(new_fn, 'w') as f:
        for l in lines_noceiling:
            f.write(l)
        print(f'write ok :\n {new_fn}')

    ceiling_fn = fn.replace('house.obj', 'ceiling.obj')
    with open(ceiling_fn, 'w') as f:
        for l in lines_ceiling:
            f.write(l)

    src_dir = os.path.dirname(fn)
    hn = os.path.basename(src_dir)
    dst_dir = os.path.join('/home/z/SUNCG/suncg_v1/parsed', hn)
    if os.path.exists(dst_dir):
      rmtree(dst_dir)
    copytree(src_dir, dst_dir)
    pass

if __name__ == '__main__':
    folder = '/home/z/SUNCG/suncg_v1/parsed'
    folder = '/DS/SUNCG/suncg_v1/parsed'
    for hn in SceneSamples.complex_structures:
      #house_name = '0f49e723572f3f03e2c9a288599b9f12'
      fn = f'{folder}/{hn}/house.obj'
      read_obj(fn)
