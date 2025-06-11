from .arctic_dataset import ArcticDataset
from .be10_dataset import BE10Dataset
from .biomassters_dataset import BioMasstersDataset, BioMasstersWinterDataset
from .biomes_dataset import BiomesDataset, BiomesWinterDataset
from .clef_dataset import CLEFDataset, CLEFBlindDataset
from .climate_dataset import Climate4Dataset, Climate4WinterDataset
from .euforest_dataset import EUForestDataset, EUForestWinterDataset, EUForestSpDataset
from .swisssdm_dataset import SwissSDMccDataset, SwissSDMciDataset, SwissSDMldDataset
from .treesatai_dataset import TreeSatAIDataset


DATASET_DICT = {
    'arctic': (ArcticDataset, 'classification', 'macro_f1', 20),
    'bigearthnet': (BE10Dataset, 'multi_label_classification', 'micro_map', 30),
    'biomassters': (BioMasstersDataset, 'distribution_regression', 'JSDivergence', 1),
    'biomassterswinter': (BioMasstersWinterDataset, 'distribution_regression', 'JSDivergence', 1),
    'biomes': (BiomesDataset, 'classification', 'macro_f1', 10),
    'biomeswinter': (BiomesWinterDataset, 'classification', 'macro_f1', 10),
    'clef': (CLEFDataset, 'multi_label_classification', 'micro_f1', 1),
    'clefblind': (CLEFBlindDataset, 'multi_label_classification', 'micro_f1', 1),
    'climate': (Climate4Dataset, 'regression', 'r2_temp', 10),
    'climatewinter': (Climate4WinterDataset, 'regression', 'r2_temp', 10),
    'euforest': (EUForestDataset, 'multi_label_classification', 'micro_f1', 5),
    'euforestsp': (EUForestSpDataset, 'multi_label_classification', 'micro_f1', 5),
    'euforestwinter': (EUForestWinterDataset, 'multi_label_classification', 'micro_f1', 5),
    'sdmcc': (SwissSDMccDataset, 'classification', 'micro_f1', 1),
    'sdmci': (SwissSDMciDataset, 'classification', 'micro_f1', 1),
    'sdmld': (SwissSDMldDataset, 'classification', 'micro_f1', 1),
    'treesatai': (TreeSatAIDataset, 'multi_label_classification', 'weighted_f1', 5),
}

DATASET_LIST = list(DATASET_DICT.keys())
