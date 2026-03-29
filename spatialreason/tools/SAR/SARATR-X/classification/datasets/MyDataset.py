import os
import pickle
from scipy.io import loadmat
import re
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class MyDataset(DatasetBase):

    dataset_dir = "MyDataset"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_Li_MSTAR_SOC.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, "TRAIN")
            test_file = os.path.join(self.dataset_dir, "TEST")
            trainval = self.read_data(trainval_file)
            test = self.read_data(test_file)
            train, val = OxfordPets.split_trainval(trainval)
            # OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots*1, 10))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, image_dir):
        label_int = {'BMP2': 0, 'BTR70': 1, 'T72': 2, 'BTR60': 3, '2S1': 4, 'BRDM2': 5, 'D7': 6, 'T62': 7, 'ZIL131': 8, 'ZSU234': 9,
                     'Bridges': 10, 'Cargo': 11, 'CoastalLands_island': 12, 'Fishing': 13, 'LandPatches': 14, 'OtherShip': 15,
                     'SeaClutterWaves': 16, 'SeaPatches': 17, 'StrongFalseAlarms': 18, 'Tanker': 19,
                     'A220': 20, 'A330': 21, 'ARJ21': 22, 'Boeing737': 23, 'Boeing787': 24
                     }

        label_name = {'BMP2': 'BMP2, a type of Infantry Fighting Vehicle',
                      'BTR70': 'BTR70, a type of Armored Personnel Carrier',
                      'T72': 'T72, a type of Tank',
                      'BTR60': 'BTR60, a type of Armored Personnel Carrier',
                      '2S1': '2S1, a type of Self-propelled Artillery',
                      'BRDM2': 'BRDM2, a type of Amphibious Armored Scout Car',
                      'D7': 'D7, a type of Bulldozer',
                      'T62': 'T62, a type of Tank',
                      'ZIL131': 'ZIL131, a type of Truck',
                      'ZSU234': 'ZSU234, a type of Self-propelled Anti-aircraft Gun',
                      'Bridges': 'Bridges', 'Cargo': 'Cargo', 'CoastalLands_island': 'CoastalLands_island',
                      'Fishing': 'Fishing', 'LandPatches': 'LandPatches', 'OtherShip': 'OtherShip',
                      'SeaClutterWaves': 'SeaClutterWaves', 'SeaPatches': 'SeaPatches',
                      'StrongFalseAlarms': 'StrongFalseAlarms', 'Tanker': 'Tanker','A220': 'A220', 'A330': 'A330', 'ARJ21': 'ARJ21', 'Boeing737': 'Boeing737',
                      'Boeing787': 'Boeing787'
                      }

        items = []

        for root, dirs, files in os.walk(image_dir):
            files = sorted(files)
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    impath = os.path.join(root, file)
                    idx = re.split('[/\\\]', impath).index('MyDataset')
                    label = label_int[re.split('[/\\\]', impath)[idx+2]]
                    classname = label_name[re.split('[/\\\]', impath)[idx+2]]
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)
        return items
