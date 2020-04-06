# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from data3d.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize

LOAD_PRED = 0

def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        pcl = batch['x']
        pcl[1] = pcl[1].to(device)
        targets = batch['y']
        pcl_ids = batch['id']

        #images, targets, image_ids = batch
        #images = images.to(device)
        with torch.no_grad():
            output = model(pcl, targets)
            output =[o.to(cpu_device) for o in output]
            for i in range(len(output)):
                output[i].constants['data_id'] = pcl_ids[i]
        results_dict.update(
            {img_id: result for img_id, result in zip(pcl_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def load_prediction(output_folder, data_loader):
    fn = os.path.join(output_folder, f"predictions.pth")
    print(fn)
    if not os.path.exists (fn):
      print('file not exist:\n'+fn)
      return None
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    predictions = torch.load(fn)
    #assert len(predictions) == len(data_loader)
    predictions = predictions[0:len(data_loader)]
    print(f'\nload {len(predictions)} predictions OK\n')
    return predictions


def inference_3d(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        iou_thresh_eval = 0.5,
        output_folder=None,
        epoch = None,
        eval_aug_thickness = None,
        load_pred = LOAD_PRED,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()

    #output_folder = output_folder + f'_{len(data_loader)}'
    if load_pred:
      predictions_load = load_prediction(output_folder, data_loader)
      if predictions_load is None:
        load_pred = False

    if load_pred:
      predictions = predictions_load
    else:
      predictions = compute_on_dataset(model, data_loader, device)
      # wait for all processes to complete before measuring the time
      synchronize()
      total_time = time.time() - start_time
      total_time_str = str(datetime.timedelta(seconds=total_time))
      logger.info(
          "Total inference time: {} ({} s / img per device, on {} devices)".format(
              total_time_str, total_time * num_devices / len(dataset), num_devices
          )
      )

      predictions = _accumulate_predictions_from_multiple_gpus(predictions)
      if not is_main_process():
          return

      if output_folder:
          torch.save(predictions, os.path.join(output_folder, f"predictions.pth"))


    extra_args = dict(
        box_only=box_only,
        #iou_types=iou_types,
        #expected_results=expected_results,
        #expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    iou_thresh_eval = iou_thresh_eval,
                    output_folder=output_folder,
                    epoch = epoch,
                    is_train = False,
                    eval_aug_thickness=eval_aug_thickness,
                    **extra_args)

