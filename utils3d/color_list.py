import numpy as np

IS_SHUFFLE = False

COLOR_LIST = np.array( [
[0,0,255],
[255,0,0],
[0,255,0],
[138,  43, 226],
[255,127,80],
[0, 255, 255],
[127,255,212],
[153,102,204],
[127,255,0],
[0,127,255],
[137,207,240],
[229,43,80],
[255,255,0],
[80,200,120],
[251,206,177],
[0,149,182],
[138,43,226],
[222,93,131],
[205,127,50],
[150,75,0],
[128,0,32],
[112,41,99],
[150,0,24],
[222,49,99],
[0,123,167],
[123,63,0],
[0,71,171],
[111,78,55],
[184,115,51],
[220,20,60],
[0,255,255],
[237,201,175],
[125,249,255],
[0,255,63],
[255,215,0],
[128,128,128],
[0,128,0],
[63,255,0],
[75,0,130],
[0,168,107],
[41,171,135],
[181,126,220],
[255,247,0],
[200,162,200],
[191,255,0],
[255,0,255],
[255,0,175],
[128,0,0],
[224,176,255],
[0,0,128],
[204,119,34],
[128,128,0],
[255,102,0],
[255,69,0],
[218,112,214],
[255,229,180],
[209,226,49],
[204,204,255],
[28,57,187],
[253,108,158],
[142,69,133],
[0,49,83],
[204,136,153],
[128,0,128],
[227,11,92],
[255,191,0],
[199,21,133],
[255,0,127],
[224,17,95],
[250,128,114],
[146,0,10],
[15,82,186],
[255,36,0],
[192,192,192],
[112,128,144],
[167,252,0],
[0,255,127],
[210,180,140],
[72,60,50],
[0,128,128],
[64,224,208],
[63,0,255],
[127,0,255],
[64,130,109],
#[255,255,255],
[0,0,0],
])

shuffle = np.array([\
        62,  6, 30, 47, 58, 59, 60, 43, 76,  7, 73,  5, 25, 75, 37, 67, 40,
        2, 56, 34, 15, 36,  4, 45, 19, 24, 48, 33,  9, 71, 64, 81, 74, 39,
        1, 66, 31, 35, 54, 18, 50, 70, 69, 27, 51, 13,  0, 42, 53, 14,  8,
       22, 26, 11, 57, 80, 28, 68, 10, 41, 52, 55, 29, 72, 49, 17, 46,  3,
       12, 21, 38, 63, 32, 20, 23, 79, 61, 44, 77, 65, 78, 16])

if IS_SHUFFLE:
    n = COLOR_LIST.shape[0]
    shuffle = np.arange(n)
    np.random.shuffle(shuffle)
    COLOR_LIST = COLOR_LIST[shuffle]

COLOR_LIST = np.tile(COLOR_LIST, [20,1]) / 255.0

import matplotlib.pyplot as plt
from skimage import io
plt.rcParams.update({'font.size': 16, 'figure.figsize': (5,5)})

def show_all():
  print(COLOR_LIST)
  print(COLOR_LIST.shape)
  n = COLOR_LIST.shape[0]
  COLOR_LIST_ = COLOR_LIST.reshape([1,-1,3])
  for i in range(0,n, 5):
    e = min(i+10, n)
    print(f'{i}:{e}')
    print(COLOR_LIST_[:,i:e]*255)
    im = plt.imshow( COLOR_LIST_[:,i:e,:], interpolation='none', aspect='auto')
    plt.colorbar(im, orientation='horizontal')
    plt.show()
    pass

def show_class_colors(classes):
  n = len(classes)
  colors = COLOR_LIST[0:n]
  k = 4
  indices = (np.arange(k*n)/k).astype(np.int)
  colors = colors[indices, :].reshape([1,-1,3])
  fig, ax = plt.subplots()
  im = ax.imshow(colors)
  ax.set_xticks(np.arange(n)*k+int(k*0.5)-1)
  ax.set_xticklabels(classes)
  ax.set_yticks([])
  plt.show()
  fig.savefig('category_colors.png')




def show___class_colors(classes):
  n = len(classes)
  COLOR_LIST_ = COLOR_LIST.reshape([1,-1,3])
  im = plt.imshow( COLOR_LIST_[:,0:n,:], interpolation='none', aspect='auto')
  plt.colorbar(im, orientation='horizontal')
  plt.show()
  pass

if __name__ == '__main__':
  show_all()

