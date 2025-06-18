from .secoeco_test_module import SeCoEcoTestModule
from .ssl4eo_test_module import SSL4EOTestModule
from .seco_test_module import SeCoTestModule
from .mocoeco_test_module import MoCoEcoTestModule
from .dofa_test_module import DOFAbaseTestModule, DOFAlargeTestModule
from .ablation_test_module import AblCalendarTestModule
from .croma_test_module import CromaTestModule
from .satlas_test_module import SatlasTestModule
from .satmae_test_module import SatMAETestModule


MODEL_DICT = {
    'ablcalendar': AblCalendarTestModule,
    'croma': CromaTestModule,
    'dofabase': DOFAbaseTestModule,
    'dofalarge': DOFAlargeTestModule,
    'mocoeco': MoCoEcoTestModule,
    'satlas': SatlasTestModule,
    'satmae': SatMAETestModule,
    'seco': SeCoTestModule,
    'secoeco': SeCoEcoTestModule,
    'ssl4eo': SSL4EOTestModule,
}

MODEL_LIST = list(MODEL_DICT.keys())
