import logging

from .suncg_eval import do_suncg_evaluation


def suncg_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("suncg evaluation doesn't support box_only, ignored.")
    logger.info("performing suncg evaluation, ignored iou_types.")
    return do_suncg_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
