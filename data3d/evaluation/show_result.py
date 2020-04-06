import open3d
import os, torch
import numpy as np
from data3d.evaluation.suncg.suncg_eval import show_pred, draw_recall_precision_score

RES_PATH0 = '/home/z/Research/Detection_3D/RES/res_sw4c_fpn432_bs1_lr5_T6655/inference_3d/paper_suncg_test_1605_iou_3_augth_2'
RES_PATH1 = '/home/z/Research/Detection_3D/RES/res_CiFl_Fpn21_bs1_lr2_T5223/inference_3d/suncg_test_1309_iou_3_augth_2'

RES_PATH = '/home/z/Research/Detection_3D/RES/res_3G6c_Fpn4321_bs1_lr5_T5223/inference_3d/suncg_test_1309_iou_3_augth_2'
RES_PATH = '/home/z/Research/Detection_3D/RES/res_3g6c_Fpn4321_bs1_lr5_Tr5227_CA/inference_3d/suncg_test_48_iou_3_augth_2'
RES_PATH = '/home/z/Research/Detection_3D/RES/res_3g6c_Fpn4321_bs1_lr5_Tr5227_CA/inference_3d/suncg_test_2_iou_3_augth_2'

Two_Res_Path = [RES_PATH0, RES_PATH1]

def show_prediction():
  pred_fn = os.path.join(RES_PATH, 'predictions.pth')
  pred_boxlists = torch.load(pred_fn)
  for preds in pred_boxlists:
    preds = preds.remove_low('scores', 0.5)
    select_ids = 1
    if select_ids:
      ids = [1,2,3]
      #ids = [5]
      preds = preds.select_by_labels(ids, 'labels')
      preds.show()

def show_prediction_gt():
  pred_fn = os.path.join(RES_PATH, 'predictions.pth')
  gt_boxlists_, pred_boxlists_, files = torch.load(pred_fn)
  show_pred(gt_boxlists_, pred_boxlists_, files)

def show_performance():
  pred_fn = os.path.join(RES_PATH, 'performance_res.pth')
  result = torch.load(pred_fn)
  result['rec_prec_score_iou_org'] = smooth_curve (result['rec_prec_score_iou_org'])
  draw_recall_precision_score(result, RES_PATH)
  #draw_recall_precision_score(result, RES_PATH, flag='10steps')


def show_performance_of_two():
  pred_fn0 = os.path.join(Two_Res_Path[0], 'performance_res.pth')
  result0 = torch.load(pred_fn0)
  pred_fn1 = os.path.join(Two_Res_Path[1], 'performance_res.pth')
  result1 = torch.load(pred_fn1)

  rec_prec_score_iou_org_0 = result0['rec_prec_score_iou_org']
  rec_prec_score_iou_org_1 = result1['rec_prec_score_iou_org']
  rec_prec_score_iou_org_0 = smooth_curve (rec_prec_score_iou_org_0)
  rec_prec_score_iou_org_1 = smooth_curve (rec_prec_score_iou_org_1)
  label_2_class0 = result0['label_2_class']
  label_2_class1 = result1['label_2_class']

  rec_prec_score_iou_org = rec_prec_score_iou_org_0 + rec_prec_score_iou_org_1[1:]
  label_2_class= {0: 'background', 1: 'wall', 2: 'window', 3: 'door', 4: 'floor', 5: 'ceiling'}


  result = {'label_2_class': label_2_class, 'rec_prec_score_iou_org':rec_prec_score_iou_org}
  RES_PATH = Two_Res_Path[0]
  draw_recall_precision_score(result, RES_PATH, smoothed=True)

  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass


def smooth_curve(rec_prec_score_iou_org, iou_threshold=0.3):
  new_rpsi = [rec_prec_score_iou_org[0]]
  num_classes = len(rec_prec_score_iou_org)
  for l in range(1, num_classes):
    rec = rec_prec_score_iou_org[l][:,0]
    psis = []
    for r in np.arange(0.0, 1.1, 0.002):
        if np.sum(rec > r) == 0:
          psi = np.array([r,0,0,0]).reshape([1,-1])
        else:
          tmp = np.nan_to_num( rec_prec_score_iou_org[l][:,1:][ rec >= r ] )
          psi = tmp.max(0).reshape([1,-1])
          iou_mask = tmp[:,2] > iou_threshold
          psi[0,2] = tmp[:,2][iou_mask].mean(0)
          psi = np.concatenate([np.array(r).reshape([1,1]), psi], 1)
        pass
        psis.append(psi)
    psis = np.concatenate(psis, 0)
    new_rpsi.append(psis)
  return new_rpsi


if __name__ == '__main__':
  show_prediction()
  #show_performance()
  #show_performance_of_two()

