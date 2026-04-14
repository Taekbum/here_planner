import mmengine
from src.utils.general_utils import InfoPrinter

def init_SLAM_model(main_cfg: mmengine.Config, info_printer: InfoPrinter):

    if main_cfg.slam.method == "coslam":
        info_printer("Initialize Co-SLAM...", 0, "Co-SLAM")
        from src.slam.coslam.coslam import CoSLAMHere as CoSLAM
        slam = CoSLAM(main_cfg, info_printer)
    else:
        assert False, f"SLAM choices: [coslam]. Current option: [{main_cfg.slam.method}]"
    return slam