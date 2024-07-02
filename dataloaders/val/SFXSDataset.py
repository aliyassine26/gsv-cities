# https://github.com/aliyassine26/gsv-cities/

from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import sys

MAIN_PATH = Path(__file__).resolve().parent.parent.parent / "utils"
sys.path.append(str(MAIN_PATH))
from config import GT_ROOT, SF_XS_PATH # type: ignore
DATASET_ROOT = SF_XS_PATH

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(
        f"Please make sure the path {DATASET_ROOT} to SF XS dataset is correct"
    )

if not path_obj.joinpath("database") or not path_obj.joinpath("queries"):
    raise Exception(
        f"Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}"
    )


class SFXSDataset(Dataset):
    def __init__(self, which_ds="sfxs_val", input_transform=None):
        """
        Initializes a new instance of the SFXSDataset class.

        Args:
            which_ds (str, optional): The dataset to use. Defaults to "sfxs_val".
                Must be either "sfxs_val" or "sfxs_test".
            input_transform (callable, optional): A function that takes a PIL Image and
                returns a transformed version. Defaults to None.

        Raises:
            AssertionError: If which_ds is not "sfxs_val" or "sfxs_test".


            - input_transform (callable): The input transform function.
            - dbImages (numpy.ndarray): An array of reference images.
            - qImages (numpy.ndarray): An array of query images.
            - ground_truth (numpy.ndarray): An array of ground truth for queries.
            - images (numpy.ndarray): An array of concatenated reference and query images.
            - num_references (int): The number of reference images.
            - num_queries (int): The number of query images.

        """


        assert which_ds.lower() in ["sfxs_val", "sfxs_test"]

        self.input_transform = input_transform

        # reference images namesf
        self.dbImages = np.load(GT_ROOT + f"SF_XS/{which_ds}_dbImages.npy")

        # query images names
        self.qImages = np.load(GT_ROOT + f"SF_XS/{which_ds}_qImages.npy")

        # ground truth
        self.ground_truth = np.load(
            GT_ROOT + f"SF_XS/{which_ds}_gtImages.npy", allow_pickle=True
        )

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    dataset = SFXSDataset("sfxs_val")
    print(len(dataset.dbImages))
    print(len(dataset.qImages))
    print(len(dataset.ground_truth))
