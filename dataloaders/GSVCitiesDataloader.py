from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torch.utils.data import Sampler
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
from dataloaders.val.SFXSDataset import SFXSDataset
from dataloaders.val.TokyoXSDataset import TokyoXSDataset
import typing
from prettytable import PrettyTable
import pytorch_lightning as pl


IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

TRAIN_CITIES = [
    "Bangkok",
    "BuenosAires",
    "LosAngeles",
    "MexicoCity",
    "OSL",  # refers to Oslo
    "Rome",
    "Barcelona",
    "Chicago",
    "Madrid",
    "Miami",
    "Phoenix",
    "TRT",  # refers to Toronto
    "Boston",
    "Lisbon",
    "Medellin",
    "Minneapolis",
    "PRG",  # refers to Prague
    "WashingtonDC",
    "Brussels",
    "London",
    "Melbourne",
    "Osaka",
    "PRS",  # refers to Paris
]


class GSVCitiesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        img_per_place: int = 4,
        min_img_per_place: int = 4,
        shuffle_all: bool = False,
        image_size: tuple = (480, 640),
        num_workers: int = 4,
        show_data_stats: bool = True,
        cities: list = TRAIN_CITIES,
        mean_std: dict = IMAGENET_MEAN_STD,
        batch_sampler: Sampler = None,
        random_sample_from_each_place: bool = True,
        val_set_names: list = ["sfxs_val"],
        test_set_names: list = ["tokyoxs_test", "sfxs_test"],
    ):
        """
        Initializes the data loader with the specified parameters.

        Parameters:
            batch_size (int): The batch size for the data loader.
            img_per_place (int): The number of images per place.
            min_img_per_place (int): The minimum number of images per place.
            shuffle_all (bool): Whether to shuffle all the data.
            image_size (tuple): The size of the images (height, width).
            num_workers (int): The number of workers for data loading.
            show_data_stats (bool): Whether to show data statistics.
            cities (list): The list of cities to include in the dataset.
            mean_std (dict): Dictionary containing 'mean' and 'std' keys for dataset normalization.
            batch_sampler: The batch sampler for the data loader.
            random_sample_from_each_place (bool): Whether to randomly sample from each place.
            val_set_names (list): The list of validation set names.
        """
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.cities = cities
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names
        self.test_set_names = test_set_names
        self.save_hyperparameters()

        self.train_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.valid_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.train_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": self.shuffle_all,
            # "persistent_workers": True,
        }

        self.valid_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers // 2,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": False,
            # "persistent_workers": True,
        }

    def setup(self, stage=None) -> None:
        """
        Setup function to prepare the data loaders and validation sets based on the stage.
        """
        if stage == "fit" or stage is None:
            # load train dataloader with reload routine
            self.reload()

            # load validation sets
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if "sfxs" in valid_set_name.lower():
                    self.val_datasets.append(
                        SFXSDataset(
                            which_ds=valid_set_name,
                            input_transform=self.valid_transform,
                        )
                    )
                else:
                    print(
                        f"Validation set {valid_set_name} does not exist or has not been implemented yet"
                    )
                    raise NotImplementedError

        if stage == "fit" or stage == "test" or stage is None:
            # load test sets
            self.test_datasets = []
            for test_set_name in self.test_set_names:
                if "sfxs" in test_set_name.lower():
                    self.test_datasets.append(
                        SFXSDataset(
                            which_ds=test_set_name,
                            input_transform=self.test_transform,
                        )
                    )
                elif "tokyoxs" in test_set_name.lower():
                    self.test_datasets.append(
                        TokyoXSDataset(
                            input_transform=self.test_transform,
                        )
                    )
                else:
                    print(
                        f"Test set {test_set_name} does not exist or has not been implemented yet"
                    )
                    raise NotImplementedError

        if self.show_data_stats:
            self.print_stats(stage)

    def reload(self) -> None:
        """
        Reloads the dataset for training with the specified parameters.
        """
        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Generate the training data loader for the model.
        """
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        """
        Generate a validation dataloader for each validation dataset using the specified configuration.
        """
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(dataset=val_dataset, **self.valid_loader_config)
            )
        return val_dataloaders

    def test_dataloader(self):
        """
        Generate a test dataloader for each test dataset using the specified configuration.
        """
        test_dataloaders = []
        for test_dataset in self.test_datasets:
            test_dataloaders.append(DataLoader(dataset=test_dataset))
        return test_dataloaders

    def print_stats(self, stage) -> None:
        """
        Generate statistics and print them in a tabular format for the training and validation datasets.
        """
        if stage == "fit":
            print()  # print a new line
            table = PrettyTable()
            table.field_names = ["Data", "Value"]
            table.align["Data"] = "l"
            table.align["Value"] = "l"
            table.header = False
            table.add_row(["# of cities", f"{len(TRAIN_CITIES)}"])
            table.add_row(["# of places", f"{self.train_dataset.__len__()}"])
            table.add_row(["# of images", f"{self.train_dataset.total_nb_images}"])
            print(table.get_string(title="Training Dataset"))
            print()

            table = PrettyTable()
            table.field_names = ["Data", "Value"]
            table.align["Data"] = "l"
            table.align["Value"] = "l"
            table.header = False
            for i, val_set_name in enumerate(self.val_set_names):
                table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
                table.add_row(["# of Queries", f"{self.val_datasets[i].num_queries}"])
                table.add_row(
                    ["# of References", f"{self.val_datasets[i].num_references}"]
                )
            print(table.get_string(title="Validation Datasets"))
            print()

        if stage == "test" or stage == "fit":
            table = PrettyTable()
            table.field_names = ["Data", "Value"]
            table.align["Data"] = "l"
            table.align["Value"] = "l"
            table.header = False
            for i, test_set_name in enumerate(self.test_set_names):
                table.add_row([f"Test set {i+1}", f"{test_set_name}"])
                table.add_row(["# of Queries", f"{self.test_datasets[i].num_queries}"])
                table.add_row(
                    ["# of References", f"{self.test_datasets[i].num_references}"]
                )
                table.add_row(["", ""])
            print(table.get_string(title="Test Datasets"))
            print()

            table = PrettyTable()
            table.field_names = ["Data", "Value"]
            table.align["Data"] = "l"
            table.align["Value"] = "l"
            table.header = False
            table.add_row(
                ["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"]
            )
            # table.add_row(
            #     ["# of iterations",
            #         f"{self.train_dataset.__len__() // self.batch_size}"]
            # )
            table.add_row(["Image size", f"{self.image_size}"])
            print(table.get_string(title="Training config"))


if __name__ == "__main__":
    dm = GSVCitiesDataModule()
    dm.setup("fit")
    # dm.print_stats()
    # train_loader = dm.train_dataloader()
    # val_loader = dm.val_dataloader()
    print("Finished")
