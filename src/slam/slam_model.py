import mmengine

from src.utils.general_utils import InfoPrinter

class SlamModel():
    def __init__(self, 
                 main_cfg: mmengine.Config,
                 info_printer: InfoPrinter
                 ) -> None:
        
        self.main_cfg = main_cfg
        self.slam_cfg = main_cfg.slam
        self.info_printer = info_printer

    def update_step(self, step):
        self.step = step
