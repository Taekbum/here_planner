import mmengine

from src.utils.general_utils import InfoPrinter

def init_simulator(main_cfg: mmengine.Config, info_printer: InfoPrinter):
    from src.simulator.habitat_simulator import HabitatSim
    sim = HabitatSim(
        main_cfg,
        info_printer
        )
    
    return sim