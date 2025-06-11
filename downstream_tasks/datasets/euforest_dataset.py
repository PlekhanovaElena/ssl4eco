import numpy as np
import os

from downstream_tasks.utils.root import get_root
from downstream_tasks.datasets.base import BaseMultiLabelClassificationDataset

LABELS = [
   "Abies","Acer","Alnus","Betula","Carpinus","Castanea","Cornus","Corylus",
   "Crataegus","Eucalyptus","Fagus","Frangula","Fraxinus","Ilex","Juglans",
   "Juniperus","Larix","Malus","Ostrya","Picea","Pinus","Populus","Prunus",
   "Pseudotsuga","Pyrus","Quercus","Robinia","Salix","Sorbus","Tilia","Ulmus"
    ] #31

SPECIES = [
    "Picea abies","Pinus sylvestris","Fagus sylvatica","Quercus robur",
    "Fraxinus excelsior","Carpinus betulus","Betula pendula","Quercus petraea",
    "Acer pseudoplatanus","Betula pubescens","Sorbus aucuparia",
    "Castanea sativa","Abies alba","Populus tremula","Alnus glutinosa",
    "Pseudotsuga menziesii","Larix decidua","Prunus avium","Salix caprea",
    "Alnus incana","Acer campestre","Pinus pinaster","Quercus pubescens",
    "Corylus avellana","Picea sitchensis","Quercus ilex","Larix kaempferi",
    "Robinia pseudoacacia","Tilia cordata","Quercus cerris","Acer platanoides",
    "Sorbus aria","Pinus nigra","Fraxinus ornus","Sorbus torminalis",
    "Frangula alnus","Prunus padus","Populus nigra","Crataegus monogyna",
    "Ulmus minor","Quercus rubra","Quercus pyrenaica","Ostrya carpinifolia",
    "Pinus halepensis","Quercus suber","Tilia platyphyllos",
    "Eucalyptus camaldulensis","Pinus contorta","Ilex aquifolium","Ulmus glabra",
    "Pinus radiata","Quercus faginea","Acer opalus","Fraxinus angustifolia",
    "Salix alba","Prunus serotina","Pinus pinea","Abies grandis",
    "Salix atrocinerea","Abies procera","Populus alba","Pyrus pyraster",
    "Pinus mugo","Juglans nigra"
] #64


class EUForestDataset(BaseMultiLabelClassificationDataset):
    index_file = get_root() + "/index_files/sdm/genus_species_labels.csv"
    image_dir = get_root() + "/sdm/euf/imgs"
    num_dim = 31
    patch_size = 108

    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        return [
            self.get_multihot(labels.split(', '))
            for labels in list(df["genus"])]

    @staticmethod
    def get_multihot(labels):
        target = np.zeros((len(LABELS),), dtype=np.int64)
        for label in labels:
            target[LABELS.index(label)] = 1
        return target


class EUForestSpDataset(BaseMultiLabelClassificationDataset):
    index_file = get_root() + "/index_files/sdm/genus_species_labels.csv"
    image_dir = get_root() + "/sdm/euf/imgs"
    num_dim = 64
    patch_size = 108

    def _read_labels(self, df):
        """Extract labels from a DataFrame."""
        return [
            self.get_multihot(labels.split(', '))
            for labels in list(df["species"])]

    @staticmethod
    def get_multihot(labels):
        target = np.zeros((len(SPECIES),), dtype=np.int64)
        for label in labels:
            target[SPECIES.index(label)] = 1
        return target


class EUForestWinterDataset(EUForestDataset):
    image_dir = get_root() + "/sdm/euf_winter/imgs"
