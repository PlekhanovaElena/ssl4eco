from .secoeco_test_module import SeCoEcoTestModule
from .ssl4eo_test_module import SSL4EOTestModule
from .seco_test_module import SeCoTestModule
from .mocoeco_test_module import MoCoEcoTestModule
from .dofa_test_module import DOFAbaseTestModule, DOFAlargeTestModule
from .ablation_test_module import AblB9TestModule, AblB9weightsTestModule, AblB12TestModule, AblB12NTestModule, AblCalendarTestModule, AblB12NweightsTestModule, AblB12weightsTestModule
from .croma_test_module import CromaTestModule
from .satlas_test_module import SatlasTestModule
from .satmae_test_module import SatMAETestModule


MODEL_DICT = {
    'ablb9': AblB9TestModule,
    'ablb9weights': AblB9weightsTestModule,
    'ablb12': AblB12TestModule,
    'ablb12n': AblB12NTestModule,
    'ablb12weights': AblB12weightsTestModule,
    'ablb12nweights': AblB12NweightsTestModule,
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
