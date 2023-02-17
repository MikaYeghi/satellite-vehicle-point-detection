from detectron2.evaluation import COCOEvaluator

class COCOPointEvaluator(COCOEvaluator):
    def __init__(self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,
    ):
        super().__init__(
            dataset_name,
            tasks=tasks,
            distributed=distributed,
            output_dir=output_dir,
            max_dets_per_image=max_dets_per_image,
            use_fast_impl=use_fast_impl,
            kpt_oks_sigmas=kpt_oks_sigmas,
            allow_cached_coco=allow_cached_coco,
        )