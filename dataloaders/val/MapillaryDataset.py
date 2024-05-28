from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
MAIN_PATH = Path(__file__).resolve().parent.parent.parent / "utils"

sys.path.append(str(MAIN_PATH))
from config import GT_ROOT, SF_XS_PATH, MPS_PATH

# NOTE: you need to download the Nordland dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench
# the folders named ref and query should reside in DATASET_ROOT path
# I hardcoded the image names and ground truth for faster evaluation
# performance is exactly the same as if you use VPR-Bench.
DATASET_ROOT = MPS_PATH

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(
        f"Please make sure the path {DATASET_ROOT} to SF XS dataset is correct"
    )

if not path_obj.joinpath("database") or not path_obj.joinpath("queries"):
    raise Exception(
        f"Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}"
    )

class MSLS(Dataset):
    def __init__(self, input_transform = None):
        

        self.input_transform = input_transform

        self.dbImages = np.load(GT_ROOT+'msls_val/msls_val_dbImages.npy')
        self.qIdx = np.load(GT_ROOT+'msls_val/msls_val_qIdx.npy')
        self.qImages = np.load(GT_ROOT+'msls_val/msls_val_qImages.npy')
        self.ground_truth = np.load(GT_ROOT+'msls_val/msls_val_pIdx.npy', allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages[self.qIdx])
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)