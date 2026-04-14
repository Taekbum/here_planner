import mmengine

from src.utils.general_utils import InfoPrinter
from src.slam.coslam.coslam import CoSLAMHere

def init_planner(main_cfg: mmengine.Config, info_printer: InfoPrinter, slam: CoSLAMHere, bound):
    if main_cfg.planner.method == "here":
        from src.planner.tare_planner import TarePlanner as TarePlanner
        planner = TarePlanner(main_cfg, info_printer, slam, bound)
    else:
        assert False, f"Planner choices: [here]. Current option: [{main_cfg.planner.method}]"
    return planner