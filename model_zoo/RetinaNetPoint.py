from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet

@META_ARCH_REGISTRY.register()
class RetinaNetPoint(RetinaNet):
    def __init__(self, cfg):
        super().__init__(cfg)