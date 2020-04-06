#u  A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_suncg.py)
from __future__ import division
import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D, merge_by_corners
from maskrcnn_benchmark.structures.boxlist_ops_3d import boxlist_iou_3d
import matplotlib.pyplot as plt
from cycler import cycler
import torch
from utils3d.color_list import COLOR_LIST
from utils3d.bbox3d_ops import Bbox3D

plt.rcParams.update({'font.size': 18, 'figure.figsize': (5,5)})

DEBUG = 1
SHOW_PRED = DEBUG and  0
DRAW_RECALL_PRECISION = DEBUG and 0
SHOW_FILE_NAMES = DEBUG and False

DRAW_REGRESSION_IOU = 0

ONLY_SAVE_NO_SHOW = 0

DEBUG_DATA_SAMPLE = 0
MERGE_BY_CORNERS = 1

def get_obj_nums(gt_boxlists, dset_metas):
    batch_size = len(gt_boxlists)
    obj_gt_nums = defaultdict(list)
    obj_gt_cum_nums = defaultdict(list)

    # switch ceilign and floor label
    label_2_class = dset_metas.label_2_class
    class_2_label = dset_metas.class_2_label
    if 'ceiling' in class_2_label and 'floor' in class_2_label:
        label_2_class[ class_2_label['floor'] ] = 'ceiling'
        label_2_class[ class_2_label['ceiling'] ] = 'floor'

    for bi in range(batch_size):
        labels = gt_boxlists[bi].get_field('labels').cpu().data.numpy()
        for l in range(dset_metas.num_classes):
            obj = label_2_class[int(l)]
            obj_gt_nums[obj].append( sum(labels==l) )
            obj_gt_cum_nums[obj].append( sum(labels<l) )
    return obj_gt_nums, obj_gt_cum_nums

def do_suncg_evaluation(dataset, predictions, iou_thresh_eval, output_folder, logger, epoch=None, is_train=None, eval_aug_thickness=None, score_threshold=0.7):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    logger.info(f'\n\nis_train: {is_train}\n')
    logger.info(f'iou_thresh: {iou_thresh_eval}\n')
    if eval_aug_thickness is not None:
      at = eval_aug_thickness['target_Y']
      logger.info(f'aug_thickness: {at}\n')
    if sum([len(p) for p in predictions]) == 0:
      print('\n\n\tno predictions to evaluate\n\n')
      return

    dset_metas = dataset.dset_metas
    pred_boxlists = []
    gt_boxlists = []
    image_ids = []
    fns = []
    for i, prediction in enumerate(predictions):
        if DEBUG_DATA_SAMPLE and i not in list(range(40,50)):
          continue
        pred_boxlists.append(prediction)
        image_id = prediction.constants['data_id']
        fns.append( dataset.files[image_id] )
        image_ids.append(image_id)
        img_info = dataset.get_img_info(image_id)
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)

    #gt_boxlists[0].show_by_labels([1])
    if SHOW_FILE_NAMES:
        print(f'\n{fns}')
    gt_nums = [len(g) for g in gt_boxlists]
    pred_nums = [len(p) for p in pred_boxlists]
    gt_num_totally = sum(gt_nums)
    if gt_num_totally == 0:
        print(f'\ngt_num_totally=0, abort evalution\n')
        return

    result = eval_detection_suncg(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=iou_thresh_eval,
        dset_metas = dset_metas,
        use_07_metric=True,
        eval_aug_thickness=eval_aug_thickness,
        score_threshold = score_threshold,
    )

    obj_gt_nums, obj_gt_cum_nums = get_obj_nums(gt_boxlists, dset_metas)
    if len(result['pred_for_each_gt']) == 0:
      print('\nno pred for each gt\n')
      return
    regression_res, missed_gt_ids, multi_preds_gt_ids, good_pred_ids, gt_ids_for_goodpreds, small_iou_preds = \
        parse_pred_for_each_gt(result['pred_for_each_gt'], obj_gt_nums, obj_gt_cum_nums, logger, iou_thresh_eval, output_folder)

    recall_precision_score_iou_10steps = result["recall_precision_score_iou_10steps"]
    result_str = performance_str(result, dataset, regression_res)
    logger.info(result_str)

    result['label_2_class'] = dset_metas.label_2_class

    if output_folder:
        dn = len(predictions)
        if epoch is not None:
          result_str = f'\nepoch: {epoch}\ndata number: {dn}\n' +  result_str
        tm_mc = 'merge_corner' if MERGE_BY_CORNERS else 'not_merge_corner'
        res_fn = os.path.join(output_folder, f"result_{dn}_{tm_mc}.txt")
        with open(res_fn, "a") as fid:
            fid.write(f'\n\niou_thresh: {iou_thresh_eval}\n')
            fid.write(result_str)
            print('write ok:\n' + res_fn + '\n')

    ap = result['ap'][1:]
    if np.isnan(ap).all():
        return result
    gt_boxlists_ = modify_gt_labels(gt_boxlists, missed_gt_ids, multi_preds_gt_ids, gt_nums, obj_gt_nums, dset_metas)
    pred_boxlists_ = modify_pred_labels(pred_boxlists, good_pred_ids, pred_nums, dset_metas, gt_ids_for_goodpreds)
    files = [dataset.files[i] for i in image_ids]
    #save_preds(gt_boxlists_, pred_boxlists_, files, output_folder)
    if SHOW_PRED:
        show_pred(gt_boxlists_, pred_boxlists_, files)

    if DRAW_RECALL_PRECISION:
        draw_recall_precision_score(result, output_folder)
    save_perform_res(result, output_folder)
    return result

def save_preds(gt_boxlists_, pred_boxlists_, files, output_folder):
  if len(gt_boxlists_) > 10:
    return
  pred_fn = os.path.join(output_folder, 'preds.pth')
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  torch.save([  gt_boxlists_, pred_boxlists_, files ], pred_fn)

def save_perform_res(result, output_folder):
  res_fn = os.path.join(output_folder, 'performance_res.pth')
  torch.save(result, res_fn)

def show_pred(gt_boxlists_, pred_boxlists_, files):
        SHOW_SMALL_IOU = False
        print('SHOW_PRED')
        for i in range(len(pred_boxlists_)):
            print(f'\n{files[i]}\n')
            #pcl_i = dataset[image_ids[i]]['x'][1][:,0:6]
            pcl_i = torch.load(files[i])[0][:,0:6]
            preds = pred_boxlists_[i].remove_low('scores', 0.1)
            #preds = pred_boxlists_[i] # already post processed in:

            select_ids = 0
            if select_ids:
              ids = [1,2,3]
              #ids = [4]
              preds = preds.select_by_labels(ids, 'labels_org')
              gt_boxlists_[i] = gt_boxlists_[i].select_by_labels(ids, 'labels_org')

              if MERGE_BY_CORNERS:
                preds = merge_by_corners(preds)

            # ~/Research/Detection_3D/maskrcnn_benchmark/modeling/roi_heads/box_head_3d/inference.py
            # cfg.MODEL.ROI_HEADS.SCORE_THRESH
            xyz_max = pcl_i[:,0:3].max(0)
            xyz_min = pcl_i[:,0:3].min(0)
            xyz_size = xyz_max - xyz_min
            pcl_i[:,0:3] -= xyz_min.reshape([1,3])
            print(f'xyz_size:{xyz_size}')

            #preds.show__together(gt_boxlists_[i], points=None, offset_x=xyz_size[0]+0.3, twolabels=False)
            #preds.show__together(gt_boxlists_[i], points=pcl_i, offset_x=xyz_size[0]+2.2, twolabels=False, mesh=0, points_keep_rate=0.9, points_sample_rate=1.0, random_color=False)


            compare_instances_with_offset = True
            if compare_instances_with_offset:
              gt_ids = preds.get_field('gt_ids').cpu().data.numpy().astype(np.int)+1

              if gt_ids.size>0 and select_ids:
                base = np.min(gt_ids[ gt_ids > 0]) - 1
                gt_ids[gt_ids > 0] -= base

              pred_colors = COLOR_LIST[gt_ids].copy()
              gt_colors = COLOR_LIST[1:len(gt_boxlists_[i])+1].copy()
              err_gt_ids = torch.nonzero(gt_boxlists_[i].get_field('labels')==0)[:,0].data.numpy().reshape([-1])
              #gt_colors[err_gt_ids] = COLOR_LIST[0].copy()


              #preds.show(points=pcl_i, points_keep_rate=0.9, points_sample_rate=1.0, colors=pred_colors)
              #gt_boxlists_[i].show(points=pcl_i, points_keep_rate=0.9, points_sample_rate=1.0, colors=gt_colors)

              preds.show__together(gt_boxlists_[i], points=pcl_i, offset_x=xyz_size[0]+7, twolabels=False, mesh=1, points_keep_rate=0.9, points_sample_rate=0.0, colors=[pred_colors, gt_colors])

              #preds.show__together(gt_boxlists_[i],offset_x=xyz_size[0]+7, twolabels=False, mesh=0, points_keep_rate=0.9, points_sample_rate=1.0, colors=[pred_colors, gt_colors])
              preds.show__together(gt_boxlists_[i], points=pcl_i, offset_x=xyz_size[0]+7, twolabels=False, mesh=0, points_keep_rate=0.9, points_sample_rate=0.0, colors=[pred_colors, gt_colors])
              #preds.show_pcl_corners(pcl_i)

              Bbox3D.draw_points(pcl_i, points_keep_rate=0.9, points_sample_rate=0.6)
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass




              #preds.show__together(gt_boxlists_[i], points=pcl_i, offset_x=0, twolabels=True, mesh=0, points_keep_rate=0.9, points_sample_rate=1.0)

            #p_labels_org = preds.get_field('labels_org')
            #g_labels_org = gt_boxlists_[i].get_field('labels_org')
            #if g_labels_org.max() == 5:
            #  wwd_mask_p = torch.nonzero( p_labels_org <= 3).squeeze()
            #  wwd_mask_g = torch.nonzero( g_labels_org <= 3).squeeze()
            #  cf_mask_p  = torch.nonzero( p_labels_org > 3).squeeze()
            #  cf_mask_g  = torch.nonzero( g_labels_org > 3).squeeze()

            #  preds[wwd_mask_p].show__together(gt_boxlists_[i][wwd_mask_g], points=pcl_i, offset_x=xyz_size[0]+2.2, twolabels=False, mesh=False, points_keep_rate=0.9, points_sample_rate=1.0)
            #  preds[cf_mask_p].show__together(gt_boxlists_[i][cf_mask_g], points=pcl_i, offset_x=xyz_size[0]+2.2, twolabels=False, mesh=False, points_keep_rate=0.9, points_sample_rate=1.0)

            #gt_boxlists_[i].show_by_labels([1])
            if SHOW_SMALL_IOU:
                small_iou_pred_ids = [p['pred_idx'] for p in  small_iou_preds[i]]
                small_ious = [p['iou'] for p in  small_iou_preds[i]]
                print(f'small iou preds: {small_iou_preds[i]}')
                if len(small_iou_pred_ids)>0:
                    preds.show_highlight(small_iou_pred_ids, points=pcl_i)

            #pred_boxlists[i].show_by_objectness(0.5, gt_boxlists[i])

def performance_str(result, dataset, regression_res):
    result_str = "\nmAP: {:.4f}\n".format(result["map"])
    ap = result['ap']
    recall_precision_score_iou_10steps = result["recall_precision_score_iou_10steps"]
    ave_prec_scr_iou  = recall_precision_score_iou_10steps.mean(1)[:,1:]
    pr_score_th5 = result['pr_score_th5']
    pr_score_th7 = result['pr_score_th7']

    class_num = len(ap)
    class_names = []
    rec7_precision = []
    rec9_precision = []
    rec7_score = []
    rec9_score = []

    score_thr_pre = []
    score_thr_rec = []
    ious_mean = []
    ious_std = []
    ious_min = []
    scores_mean = []
    scores_std = []
    scores_min = []
    missed_gt_rates = []
    multi_gt_rates = []
    gt_nums = []

    for i in range(class_num):
        clsn = dataset.map_class_id_to_class_name(i)
        if i==0:
            clsn = 'mean'
        class_names.append(clsn )
        rec7_precision.append( recall_precision_score_iou_10steps[i][7][1] )
        rec9_precision.append( recall_precision_score_iou_10steps[i][9][1] )
        rec7_score.append( recall_precision_score_iou_10steps[i][7][2] )
        rec9_score.append( recall_precision_score_iou_10steps[i][9][2] )
        if i==0:
            score_thr_pre.append(   np.nan )
            score_thr_rec.append(   np.nan )
            ious_mean.append( np.nan )
            ious_std.append( np.nan)
            ious_min.append( np.nan )
            scores_mean.append( np.nan )
            scores_std.append( np.nan)
            scores_min.append( np.nan )
            missed_gt_rates.append( np.nan )
            multi_gt_rates.append( np.nan )
            gt_nums.append( np.nan )
        else:
            if clsn in regression_res:
              score_thr_pre.append( regression_res[clsn]['prec_st'] )
              score_thr_rec.append( regression_res[clsn]['rec_st'] )
              ious_mean.append( regression_res[clsn]['ave_std_iou'][0] )
              ious_std.append( regression_res[clsn]['ave_std_iou'][1] )
              ious_min.append( regression_res[clsn]['min_max_iou'][0] )
              scores_mean.append( regression_res[clsn]['ave_std_score'][0] )
              scores_std.append( regression_res[clsn] ['ave_std_score'][1] )
              scores_min.append( regression_res[clsn] ['min_max_score'][0] )
              missed_gt_rates.append( regression_res[clsn]['matched_missed_multi_rate'][1] )
              multi_gt_rates.append( regression_res[clsn] ['matched_missed_multi_rate'][2] )
              gt_nums.append( regression_res[clsn] ['missed_multi_sum_gtnum'][2] )
            else:
              score_thr_pre.append(   np.nan )
              score_thr_rec.append(   np.nan )
              ious_mean.append(   np.nan )
              ious_std.append(  np.nan )
              ious_min.append(  np.nan )
              scores_mean.append(  np.nan )
              scores_std.append(  np.nan )
              scores_min.append(  np.nan )
              missed_gt_rates.append(  np.nan )
              multi_gt_rates.append(  np.nan )
              gt_nums.append(  0 )


    score_thr_pre[0] = np.mean(score_thr_pre[1:])
    score_thr_rec[0] = np.mean(score_thr_rec[1:])
    ious_mean[0] = np.mean(ious_mean[1:])
    ious_std[0] = np.mean(ious_std[1:])
    ious_min[0] = np.min(ious_min[1:])

    scores_mean[0] = np.mean(scores_mean[1:])
    scores_std[0] =  np.mean(scores_std[1:])
    scores_min[0] =  np.min( scores_min[1:])

    missed_gt_rates[0] = np.mean(missed_gt_rates[1:])
    multi_gt_rates[0] = np.mean(multi_gt_rates[1:])
    gt_nums[0] = np.mean(gt_nums[1:]).astype(np.int)


    result_str += f'{"class ":13}' + '  '.join([f'{c:<10}' for c in  class_names]) + '\\\\\n '
    result_str += '\hline\n'
    result_str += f'{"AP ":13}' + '  '.join([f'{p*100:<10.2f}' for p in ap]) + '\\\\\n'
    result_str += f'{"mIoU ":13}' + '  '.join([f'{p*100:<10.2f}' for p in ave_prec_scr_iou[:,2]]) + '\\\\\n'

    result_str += f'\n{"st5 prec ":13}' + '  '.join([f'{p[0]*100:<10.2f}' for p in pr_score_th5]) + '\\\\\n'
    result_str += f'{"st5 rec ":13}' + '  '.join([f'{p[1]*100:<10.2f}' for p in pr_score_th5]) + '\\\\\n'
    result_str += f'{"iou mean ":13}' + '  '.join([f'{p*100:<10.2f}' for p in ious_mean]) + '\\\\\n'
    result_str += f'{"iou std ":13}' + '  '.join([f'{p*100:<10.2f}' for p in ious_std]) + '\\\\\n'
    result_str += f'{"iou min ":13}' + '  '.join([f'{p*100:<10.2f}' for p in ious_min]) + '\\\\\n'

    result_str += f'\n{"s=0.5 prec ":13}' + '  '.join([f'{p*100:<10.2f}' for p in score_thr_pre]) + '\\\\\n'
    result_str += f'{"s=0.5 rec ":13}' + '  '.join([f'{p*100:<10.2f}' for p in score_thr_rec]) + '\\\\\n'

    result_str += f'\n{"st7 prec ":13}' + '  '.join([f'{p[0]*100:<10.2f}' for p in pr_score_th7]) + '\\\\\n'
    result_str += f'{"st7 rec ":13}' + '  '.join([f'{p[1]*100:<10.2f}' for p in pr_score_th7]) + '\\\\\n'
    result_str += f'{"r7p ":13}' + '  '.join([f'{p*100:<10.2f}' for p in rec7_precision]) + '\\\\\n'
    result_str += f'{"r9p ":13}' + '  '.join([f'{p*100:<10.2f}' for p in rec9_precision]) + '\\\\\n'

    result_str += f'{"r7s ":13}' + '  '.join([f'{p:<10.4f}' for p in rec7_score]) + '\\\\\n'
    result_str += f'{"r9s ":13}' + '  '.join([f'{p:<10.4f}' for p in rec9_score]) + '\\\\\n'
    result_str += f'{"score mean ":13}' + '  '.join([f'{p:<10.4f}' for p in scores_mean]) + '\\\\\n'
    result_str += f'{"score std ":13}' + '  '.join([f'{p:<10.4f}' for p in  scores_std]) + '\\\\\n'
    result_str += f'{"score min ":13}' + '  '.join([f'{p:<10.4f}' for p in  scores_min]) + '\\\\\n'
    #result_str += f'{"missed gt ":13}' + '  '.join([f'{p:<10.4f}' for p in  missed_gt_rates]) + '\\\\\n'
    result_str += f'{"multi gt ":13}' + '  '.join([f'{p:<10.4f}' for p in  multi_gt_rates]) + '\\\\\n'
    result_str += f'{"gt num ":13}' + '  '.join([f'{p:<10d}' for p in gt_nums]) + '\\\\\n'
    result_str += '\n'
    #print(result_str)
    return result_str


def modify_pred_labels(pred_boxlists, good_pred_ids, pred_nums, dset_metas, gt_ids_for_goodpreds):
    # incorrect pred: 0,  others: class label

    batch_size = len(pred_nums)
    new_pred_boxlists = []
    for bi in range(batch_size):
        labels_i_org = pred_boxlists[bi].get_field('labels')
        labels_i = np.zeros([pred_nums[bi]], dtype=np.int32)
        gtids_i = np.zeros([pred_nums[bi]], dtype=np.int32) - 1
        for obj in good_pred_ids:
            l = dset_metas.class_2_label[obj]
            if good_pred_ids[obj][bi].shape[0] > 0:
                labels_i[good_pred_ids[obj][bi]] = l
                gtids_i[good_pred_ids[obj][bi]] = gt_ids_for_goodpreds[obj][bi]

        pred = pred_boxlists[bi].copy()
        pred.add_field('labels', labels_i)
        pred.add_field('labels_org', labels_i_org)
        pred.add_field('gt_ids', gtids_i)
        new_pred_boxlists.append(pred)
    return new_pred_boxlists

def modify_gt_labels(gt_boxlists, missed_gt_ids, multi_preds_gt_ids, gt_nums, obj_gt_nums, dset_metas):
    # missed:0, matched: class label , multi:

    batch_size = len(gt_nums)
    gt_labels = []
    new_gt_boxlists = []
    for bi in range(batch_size):
        #labels_i = np.zeros([gt_nums[bi]], dtype=np.int32)
        labels_i_org = gt_boxlists[bi].get_field('labels')
        labels_i = labels_i_org.clone().detach()
        #labels_i = np.random.choice(gt_nums[bi], gt_nums[bi], replace=False)+2
        start = 0 # the gt_ids is only of one class (TAG: GT_MASK)
        for obj in missed_gt_ids:
            #gt_label_i = dset_metas.class_2_label[obj]
            labels_i[ missed_gt_ids[obj][bi] + start ] = 0
            #labels_i[ multi_preds_gt_ids[obj][bi] + start ] = dset_metas.label_num()
            start += obj_gt_nums[obj][bi]
        gt_labels.append(labels_i)

        boxlist = gt_boxlists[bi].copy()
        boxlist.add_field('labels', labels_i)
        boxlist.add_field('labels_org', labels_i_org)
        new_gt_boxlists.append(boxlist)

    return new_gt_boxlists

def parse_pred_for_each_gt(pred_for_each_gt, obj_gt_nums, obj_gt_cum_nums, logger, iou_thresh_eval, output_folder, score_thres=0.5):
    missed_gt_ids = defaultdict(list)
    multi_preds_gt_ids = defaultdict(list)
    ious = defaultdict(list)
    scores = defaultdict(list)
    good_pred_ids = defaultdict(list)
    score_thres_nums = defaultdict(list)
    success_nums = defaultdict(list)
    gt_ids_for_goodpreds = defaultdict(list)

    ious_flat = {}
    scores_flat = {}
    regression_res = {}
    batch_sizes = [len(v) for v in pred_for_each_gt.values()]
    assert min(batch_sizes) == max(batch_sizes)
    batch_size = batch_sizes[0]

    ious_all = defaultdict(list)
    scores_all = defaultdict(list)

    small_iou_preds = []
    for bi in range(batch_size):
        small_iou_preds.append([])
    small_iou_threshold = 0.5

    for obj in pred_for_each_gt.keys():
          for bi in range(batch_size):
                if len(pred_for_each_gt[obj]) == 0:
                    continue
                peg = pred_for_each_gt[obj][bi]

                #-------------------------------
                # get scores and iou
                ious_bi = []
                scores_bi = []
                good_pred_ids_bi = []
                gt_ids_bi = []
                for gt_id in peg:
                    # if a gt matches multiple preds, the max score one is positive
                    # here the first one actually has the max score
                    gt_ids_bi.append(gt_id)
                    peg_max_score = peg[gt_id][0]
                    scores_bi.append( peg_max_score['score'] )
                    ious_bi.append( peg_max_score['iou'] )
                    good_pred_ids_bi.append( peg_max_score['pred_idx'] )
                scores_bi = np.array(scores_bi)
                ious_bi = np.array(ious_bi)
                gt_ids_bi = np.array(gt_ids_bi)

                ious_all[obj].append(ious_bi)
                scores_all[obj].append(scores_bi)


                # (1) successful detection: (score > score_thres + iou > iou_thresh)
                score_mask = scores_bi >= score_thres
                iou_mask = ious_bi > iou_thresh_eval
                success_mask = score_mask * iou_mask

                # record for calculating recall and precision when score>0.5
                score_thres_nums[obj].append( score_mask.sum() )
                success_nums[obj].append( success_mask.sum())

                scores_bi = scores_bi[success_mask]
                ious_bi = ious_bi[success_mask]
                gt_ids_bi = gt_ids_bi[success_mask]

                ious[obj].append(ious_bi)
                scores[obj].append(scores_bi)
                good_pred_ids_bi = np.array(good_pred_ids_bi)[success_mask]
                good_pred_ids[obj].append( good_pred_ids_bi )
                base = (np.array(gt_ids_bi)>=0).astype(np.int) * obj_gt_cum_nums[obj][bi]
                gt_ids_for_goodpreds[obj].append( gt_ids_bi + base)

                #-------------------------------
                # small iou
                small_iou_preds_bi = []
                for pi in peg:
                    for peg_ in peg[pi]:
                        if peg_['score'] < score_thres:
                          continue
                        if peg_['iou'] < small_iou_threshold:
                            bad_p = {}
                            bad_p['pred_idx'] = peg_['pred_idx']
                            bad_p['iou'] = peg_['iou']
                            bad_p['score'] = peg_['score']
                            bad_p['gt_idx'] = pi
                            bad_p['class'] = obj
                            small_iou_preds_bi.append(bad_p)
                small_iou_preds[bi] += small_iou_preds_bi

                #-------------------------------
                #gt_ids is only the index inside of one single class gts: (TAG: GT_MASK)
                pred_num_each_gt = np.histogram(gt_ids_bi, bins=range(obj_gt_nums[obj][bi]+1))[0]
                pred_num_hist = np.histogram(pred_num_each_gt, bins=[0,1,2,3,4])[0]
                #print(f'{pred_num_hist[0]} gt boxes are missed \n{pred_num_hist[1]} t Boxes got one prediction')
                #print(f'{pred_num_hist[2]} gt boxes got 2 predictions')
                missed_gt_ids_bi = np.where(pred_num_each_gt==0)[0]
                multi_preds_gt_ids_bi = np.where(pred_num_each_gt>1)[0]

                missed_gt_ids[obj].append(missed_gt_ids_bi)
                multi_preds_gt_ids[obj].append(multi_preds_gt_ids_bi)

                pass

          ious_flat[obj] = np.concatenate(ious[obj], 0)
          scores_flat[obj] = np.concatenate(scores[obj], 0)

          if ious_flat[obj].shape[0] == 0:
             ious_flat[obj] = np.array(np.nan)
             scores_flat[obj] = np.array(np.nan)

          ave_iou = np.mean(ious_flat[obj])
          std_iou = np.std(ious_flat[obj])
          max_iou = np.max(ious_flat[obj])
          min_iou = np.min(ious_flat[obj])
          ave_score = np.mean(scores_flat[obj])
          std_score = np.std(scores_flat[obj])
          max_score =  np.max(scores_flat[obj])
          min_score =  np.min(scores_flat[obj])
          rec_st = 1.0* np.sum(success_nums[obj]) / np.sum(obj_gt_nums[obj])
          prec_st = 1.0* np.sum(success_nums[obj]) / np.sum(score_thres_nums[obj])

          regression_res[obj] = {}
          regression_res[obj]['min_max_iou'] = [min_iou, max_iou]
          regression_res[obj]['ave_std_iou'] = [ave_iou, std_iou]
          regression_res[obj]['min_max_score'] = [min_score, max_score]
          regression_res[obj]['ave_std_score'] = [ave_score, std_score]
          regression_res[obj]['prec_st'] = prec_st
          regression_res[obj]['rec_st'] = rec_st

          missed_gt_num = sum([len(gti) for gti in missed_gt_ids[obj] ])
          multi_gt_num = sum([len(gti) for gti in multi_preds_gt_ids[obj] ])
          gt_num_sum = sum(obj_gt_nums[obj])
          regression_res[obj]['missed_multi_sum_gtnum'] = [missed_gt_num, multi_gt_num, gt_num_sum]
          missed_rate = 1.0*missed_gt_num / gt_num_sum
          multi_rate = 1.0*multi_gt_num / gt_num_sum
          matched_rate = 1 - missed_rate - multi_rate
          regression_res[obj]['matched_missed_multi_rate'] = [matched_rate, missed_rate, multi_rate]
          pass

    reg_str = regression_res_str(regression_res)
    logger.info(reg_str)

    if DRAW_REGRESSION_IOU:
        for obj in ious_flat:
            fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
            io = ious_flat[obj]
            if len(io.shape) == 0:
              continue

            io_hist, bin_edges = np.histogram(io, bins=np.arange(11)/10.0)
            io_hist = io_hist *1.0/ io_hist.sum()
            plt.bar(bin_edges[0:-1], io_hist, width=0.1, align='edge')

            #axs.hist(io, bins=20, density=True)
            #plt.plot()
            plt.xlabel(f'iou')
            plt.ylabel('count')
            title = f'iou histogram of {obj}'
            #plt.title(title)
            fname = f'{output_folder}/iou_hist_{obj}.png'
            fig.savefig(fname)
            try:
              io0_rate = np.sum(io<0.1)/io.shape[0]
            except:
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
            print(f'\nio<0.1: {io0_rate}')
            print(fname)
            if not ONLY_SAVE_NO_SHOW:
              plt.show()
            plt.close()


            #fig = plt.figure(1)
            #s = scores_flat[obj]
            #plt.plot(s, io ,'.')
            #plt.xlabel(f'score')
            #plt.ylabel('iou')
            #fname = f'score_iou_{obj}.png'
            #fig.savefig(fname)
            #if not ONLY_SAVE_NO_SHOW:
            #  plt.show()
            #plt.close()

            pass

    return regression_res, missed_gt_ids, multi_preds_gt_ids, good_pred_ids, gt_ids_for_goodpreds, small_iou_preds

def regression_res_str(regression_res):
    reg_str = '\n\nregression result\n'
    for key in regression_res:
        value = regression_res[key]
        reg_str += f'{key}:\n{value}\n'
    return reg_str

def draw_recall_precision_score(result, output_folder, flag='', smoothed=False):
    if flag != '10steps':
      rec_prec_sco_iou_list = result['rec_prec_score_iou_org']
    else:
      rec_prec_sco_iou_list = result['recall_precision_score_iou_10steps']
    label_2_class = result['label_2_class']

    num_classes = len(rec_prec_sco_iou_list)

    default_cycler = (cycler(color=['r', 'g', 'b', 'y','k']) + cycler(linestyle=['-', '--', ':', '-.','--']))
    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=default_cycler)


    # together precision-recall
    fig = plt.figure()
    for i in range(1,num_classes):
        obj = label_2_class[i]
        rp = rec_prec_sco_iou_list[i]
        if not smoothed:
            if DEBUG:
              rp = rm_bad_head(rp)
            rp  = expand_rp_tail(rp)
        plt.plot(rp[:,0]*100, rp[:,1]*100, label=obj)
    plt.xlabel('recall (%)')
    plt.ylabel('precision (%)')
    plt.legend()
    fig_fn = output_folder + '/recall_precision.png'
    fig.savefig(fig_fn)
    print('save: '+fig_fn)
    if not ONLY_SAVE_NO_SHOW:
      plt.show()
    plt.close()

    # together iou-recall
    from scipy.signal import savgol_filter

    fig = plt.figure()
    for i in range(1,num_classes):
        obj = label_2_class[i]
        rp = rec_prec_sco_iou_list[i]
        if not smoothed:
            if DEBUG:
              rp = rm_bad_head(rp)
            rp  = expand_rp_tail(rp)
        iou_i = rp[:,3]
        #iou_i = savgol_filter(rp[:,3], 101, 5)
        plt.plot(rp[:,0]*100, iou_i*100, label=obj)
    plt.xlabel('recall (%)')
    plt.ylabel('IoU (%)')
    plt.legend()
    fig_fn = output_folder + '/recall_iou.png'
    fig.savefig(fig_fn)
    print('save: '+fig_fn)
    if not ONLY_SAVE_NO_SHOW:
      plt.show()
    plt.close()

    # together score-recall
    fig = plt.figure()
    for i in range(1,num_classes):
        obj = label_2_class[i]
        rp = rec_prec_sco_iou_list[i]
        if not smoothed:
            rp  = expand_rp_tail(rp)
        plt.plot(rp[:,0]*100, rp[:,2], label=obj)
    plt.xlabel('recall (%)')
    plt.ylabel('score')
    plt.legend()
    fig_fn = output_folder + '/recall_score.png'
    fig.savefig(fig_fn)
    print('save: '+fig_fn)
    if not ONLY_SAVE_NO_SHOW:
      plt.show()
    plt.close()

    return


    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    # separate
    for i in range(1,num_classes):
        obj = label_2_class[i]
        if i==0:
            if flag == '10steps':
                continue
            obj = 'ave'
        rp = rec_prec_sco_iou_list[i]

        if DEBUG:
          rp = rm_bad_head(rp)
        rp  = expand_rp_tail(rp)

        #print(f'\n{obj} recall - precision - score\n{rp}')
        fig = plt.figure(i)
        plt.plot(rp[:,0], rp[:,1], label='precision')
        plt.plot(rp[:,0], rp[:,2], label='score threshold')
        plt.legend()
        #plt.ylabel('precision')
        plt.xlabel('recall')
        title = flag+' '+obj+' recall-precision'
        plt.title(title)
        fig_fn = output_folder + '/' + title+'.png'
        fig.savefig(fig_fn)
        print('save: '+fig_fn)
        if not ONLY_SAVE_NO_SHOW:
          plt.show()
        plt.close()
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

def expand_rp_tail(rp):
  if rp[-1,0] < 1:
    rp_tail = np.array([[rp[-1,0]+0.01, 0,0,0], [1,0,0,0]])
    rp = np.concatenate([rp,rp_tail],0)
  return rp

def rm_bad_head(rp):
  n = int(rp.shape[0] * 0.1)
  if n < 2:
    return rp
  ma = rp[:n, 1].max()
  mi = rp[n, 1]
  for i in range(n):
    rp[i,1] = ma - 1.0*i/n*(ma-mi)
  #rp[:n,1] = np.clip(rp[:n,1], a_min=m,a_max=None)
  return rp
#def get_obejct_numbers(boxlist, dset_metas):
#    labels = boxlist.get_field('labels').data.numpy()
#    lset = list(set(labels))
#    obj_nums = {}
#    for l in lset:
#        obj_nums[dset_metas.label_2_class[l]] = sum(labels==l)
#    return obj_nums

def eval_detection_suncg(pred_boxlists, gt_boxlists, iou_thresh, dset_metas, use_07_metric=False, eval_aug_thickness=None, score_threshold=0.5):
    """Evaluate on suncg dataset.
    Args:
        pred_boxlists(list[BoxList3D]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList3D]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec, pred_for_each_gt, scores, predious = calc_detection_suncg_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh, dset_metas=dset_metas, eval_aug_thickness=eval_aug_thickness
    )
    #mious = cal_mious(rec, predious, iou_thresh, dset_metas)
    rec_prec_score_iou_org = [np.concatenate([np.array(r).reshape([-1,1]), np.array(p).reshape([-1,1]), np.array(s).reshape([-1,1]), np.array(u).reshape([-1,1])],1) \
                    for r,p,s,u in zip(rec, prec, scores, predious)]
    pr_score_th5 = pr_of_score_threshold(prec, rec, scores, 0.5)
    pr_score_th7 = pr_of_score_threshold(prec, rec, scores, 0.7)
    ap, recall_precision_score_iou_10steps = calc_detection_suncg_ap(prec, rec, scores, predious, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap), "rec_prec_score_iou_org":rec_prec_score_iou_org, "recall_precision_score_iou_10steps":recall_precision_score_iou_10steps,
            "pred_for_each_gt":pred_for_each_gt, 'pr_score_th5': pr_score_th5, 'pr_score_th7': pr_score_th7 }

def pr_of_score_threshold(prec, rec, scores, score_threshold):
    pr_score_th = [[np.nan, np.nan]]
    n = len(prec)
    for i in range(1,n):
      if scores[i] is None:
        continue
      k = np.sum(scores[i] > score_threshold) - 1
      pr_score_th.append( [prec[i][k], rec[i][k]] )
    pr_score_th = np.array(pr_score_th)
    pr_score_th[0,:] = pr_score_th[1:,:].mean(0)
    return pr_score_th

def calc_detection_suncg_prec_rec(gt_boxlists, pred_boxlists, iou_thresh, dset_metas, eval_aug_thickness):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    # The pred having maximum iou with a gt is matched with the gt.
    # If multiple preds share same maximum iou gt, the one with highest score is
    # selected. NOTICE HERE, not the one with highest iou! Because, in test,
    # only score is available.
    match = defaultdict(list)  # 1:true, 0:false, -1:ignore
    predious = defaultdict(list)

    pred_for_each_gt = defaultdict(list)
    batch_size = len(gt_boxlists)

    bi = -1
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        bi += 1
        pred_bbox = pred_boxlist.bbox3d.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox3d.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            obj_name = dset_metas.label_2_class[l]
            pred_mask_l = pred_label == l
            pred_ids_l = np.where(pred_mask_l)[0]
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            # Extract gt only of current class, thus gt_index is the index
            # inside of one signle class gts, not of all gts
            # TAG: GT_MASK
            gt_bbox_l = gt_bbox[gt_mask_l]

            n_pos[l] += gt_bbox_l.shape[0]
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                predious[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            pred_bbox_l = pred_bbox_l.copy()
            gt_bbox_l = gt_bbox_l.copy()
            iou = boxlist_iou_3d(
                BoxList3D(gt_bbox_l, gt_boxlist.size3d, gt_boxlist.mode, None, gt_boxlist.constants),
                BoxList3D(pred_bbox_l, pred_boxlist.size3d, pred_boxlist.mode, None, pred_boxlist.constants),
                aug_thickness = eval_aug_thickness,
                criterion = -1,
                flag='eval'
            ).numpy()   # [gt_nm,pred_num]

            gt_index = iou.argmax(axis=0) # the gt index for each predicion
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=0) < iou_thresh] = -1
            pred_for_each_gt_l = defaultdict(list)
            neg_count =  0
            for pi in range(gt_index.shape[0]):
                pis = {'pred_idx': pred_ids_l[pi], 'iou':iou[gt_index[pi], pi], 'score':score[l][pi]}
                gt_idx = gt_index[pi]
                if gt_idx<0:
                    neg_count += 1
                    gt_idx -= (gt_idx==-1) * neg_count
                pred_for_each_gt_l[gt_idx].append(pis)

            if obj_name not in pred_for_each_gt:
                for iii in range(batch_size):
                    pred_for_each_gt[obj_name].append(defaultdict(list))
            pred_for_each_gt[obj_name][bi] = pred_for_each_gt_l

            predious[l].extend( iou.max(0) )

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        # gt_index is already sorted by scores,
                        # thus the first pred match a gt box is set 1
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

            del iou
            pass


    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    scores = [None] * n_fg_class
    pred_ious = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        if score_l.shape[0] == 0:
          continue
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        scores[l] = score_l[order]
        match_l = match_l[order]
        predious[l] = np.array( predious[l], dtype=np.float)
        if not predious[l].shape[0] == order.shape[0]:
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
        pred_ious[l] = predious[l][order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        #if n_pos[l] > 0:
        rec[l] = tp / n_pos[l]

    #plt.plot(rec[1], label='rec')
    #plt.plot(prec[1], label='prec')
    #plt.plot(scores[1], label='score')
    #plt.legend()
    #plt.show()
    return prec, rec, pred_for_each_gt, scores, pred_ious

def calc_detection_suncg_ap(prec, rec, scores, predious, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    recall_precision_score_iou_10steps = np.empty([n_fg_class, 11, 4])
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            recall_precision_score_iou_10steps[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            rp = []
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                if np.sum(rec[l] >= t) == 0:
                    iou = 0
                else:
                    iou = np.max(np.nan_to_num(predious[l])[rec[l] >= t])
                if np.sum(rec[l] <= t) == 0:
                    try:
                      s = np.max(scores[l]) + 0.01
                    except:
                      import pdb; pdb.set_trace()  # XXX BREAKPOINT
                      pass
                else:
                    s = np.min(scores[l][rec[l] <= t])
                ap[l] += p / 11
                rp.append([t, p, s, iou]) # [recall, precision, score_thres]
            recall_precision_score_iou_10steps[l] = np.array(rp)
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    # set the first as average
    recall_precision_score_iou_10steps[0] = recall_precision_score_iou_10steps[1:].mean(0)
    ap[0] = ap[1:].mean()
    return ap, recall_precision_score_iou_10steps


def cal_mious(predious, iou_thresh, dset_metas):
  mious = [np.nan]*(1+len(predious))
  for obj in predious:
    miou = []
    bn = len( predious[obj] )
    for i in range(bn):
      mask = predious[obj][i] >iou_thresh
      miou.append( np.mean( predious[obj][i][mask]) )
    miou = np.mean(miou)

    l = dset_metas.class_2_label[obj]
    mious[l] = miou

  return mious
