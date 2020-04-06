from .suncg import suncg_evaluation


def evaluate(dataset, predictions, iou_thresh_eval,  output_folder, epoch, is_train, eval_aug_thickness, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """

    args = dict(
        dataset=dataset, predictions=predictions, iou_thresh_eval=iou_thresh_eval, output_folder=output_folder, epoch=epoch, is_train=is_train, eval_aug_thickness=eval_aug_thickness, **kwargs
    )

    return suncg_evaluation(**args)
