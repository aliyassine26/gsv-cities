# https://github.com/amaralibey/gsv-cities

import os
import sys
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parent.parent.parent / "utils"
sys.path.append(str(MAIN_PATH))

import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import typing
from torchvision import transforms as T
from config import GSV_CITIES_PATH, DF_PATH  # type: ignore



def show_image(image: str, title: str) -> None:
    """
    Display an image with a given title.

    Parameters:
        image (numpy.ndarray): The image to be displayed.
        title (str): The title of the image.

    Returns:
        None
    """
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


default_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

BASE_PATH = GSV_CITIES_PATH


class GSVBaseDataset(Dataset):
    def __init__(
        self,
        root_dir: str = BASE_PATH,
        dataframes_dir: str = DF_PATH,
        transform: T.transforms.Compose = default_transform,
    ):
        """Initialize the GSVBaseDataset class.

        Args:
            root_dir (str): root directory of the dataset.
            transform (transform, optional): transformation to apply to the images.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.dataframes_dir = dataframes_dir
        self.dataframes = self.__get_dataframes()
        self.main_df = pd.concat(list(self.dataframes.values()), axis=0)
        self.total_nb_images = len(self.main_df)

    def __get_dataframes(self) -> dict:
        """
        Get a dictionary of dataframes for each city in the dataframes directory.

        Returns:
            dict: A dictionary where the keys are the city names and the values are the corresponding dataframes.
        """
        dataframes = {}
        for city in os.listdir(self.dataframes_dir):
            if city.endswith(".csv"):
                df = pd.read_csv(os.path.join(self.dataframes_dir, city))
                dataframes[city] = df
        return dataframes

    def get_dataframes_list(self) -> list:
        """
        Get a list of dataframes from the `self.dataframes` dictionary.

        Returns:
            list: A list of dataframes.
        """
        return list(self.dataframes.values())

    def get_dataframe(self, city) -> pd.DataFrame:
        """
        Get the dataframe corresponding to the given city.

        Args:
            city (str): The name of the city.

        Returns:
            pd.DataFrame: The dataframe for the given city.
        """
        return self.dataframes[city]

    def __len__(self) -> int:
        return sum(list(v.shape[0] for k, v in self.dataframes.items()))

    def __getitem__(self, idx) -> typing.Tuple[str, str, float, float, Image.Image]:
        """
        Get an item from the dataset at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[str, str, float, float, Image.Image]: A tuple containing the place ID, class name, UTM x-coordinate, UTM y-coordinate, and the image.
        """
        place_id, class_name, UTMx, UTMy, img_path = self.main_df.iloc[idx]
        image = Image.open(
            os.path.join(self.root_dir, class_name.lower(), img_path)
        ).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return place_id, class_name, UTMx, UTMy, image

    def show_random_images(self, n=4) -> None:
        """
        Display random images from the dataset.

        This function randomly selects `n` images from the dataset and displays them.
        The images are displayed with their corresponding place ID, class name, and coordinates.

        Parameters:
            n (int, optional): The number of images to display. Defaults to 4.

        Returns:
            None
        """
        for i in range(n):
            idx = random.randint(0, self.main_df.shape[0])
            place_id, class_name, UTMx, UTMy, image = self.__getitem__(idx)
            show_image(
                image.permute(1, 2, 0),
                f"Place ID: {place_id}, Class: {class_name}, Coordinates: {(UTMx, UTMy)}",
            )

    def show_random_images_by_city(self, class_name, n=4) -> None:
        """
        Display random images from a specific city.

        This function randomly selects `n` images from the dataset for a given city and displays them.
        The images are displayed with their corresponding place ID, class name, and coordinates.

        Parameters:
            class_name (str): The name of the city.
            n (int, optional): The number of images to display. Defaults to 4.

        Raises:
            ValueError: If the city has less than `n` images.

        Returns:
            None
        """
        class_df = self.main_df[self.main_df["class_name"] == class_name]
        if class_df.shape[0] < n:
            raise ValueError(
                f"Class: {class_name} has only {class_df.shape[0]} images."
            )
        for i in range(n):
            idx = random.randint(0, class_df.shape[0])
            place_id, class_name, UTMx, UTMy, image = self.__getitem__(idx)
            show_image(
                image.permute(1, 2, 0),
                f"Place ID: {place_id}, Class: {class_name}, Coordinates: {(UTMx, UTMy)}",
            )

    def show_random_images_by_place(self, place_id, class_name, n=4) -> None:
        """
        Display random images from a specific place.

        This function randomly selects `n` images from the dataset for a given place and displays them.
        The images are displayed with their corresponding place ID, class name, and coordinates.

        Parameters:
            place_id (int): The ID of the place.
            class_name (str): The name of the class.
            n (int, optional): The number of images to display. Defaults to 4.

        Raises:
            ValueError: If the place has less than `n` images.

        Returns:
            None
        """
        place_df = self.main_df[
            (self.main_df["place_id"] == place_id)
            & (self.main_df["class_name"] == class_name)
        ]
        if place_df.shape[0] < n:
            raise ValueError(
                f"Place ID: {place_id} has only {place_df.shape[0]} images."
            )
        for i in range(n):
            idx = random.randint(0, place_df.shape[0])
            place_id, class_name, UTMx, UTMy, image = self.__getitem__(idx)
            show_image(
                image.permute(1, 2, 0),
                f"Place ID: {place_id}, Class: {class_name}, Coordinates: {(UTMx, UTMy)}",
            )

    def save_main_df(self, path) -> None:
        self.main_df.to_csv(path, index=False)


#########################################################################################################


class GSVCitiesDataset(GSVBaseDataset):
    def __init__(
        self,
        cities: list = ["London", "Boston"],
        img_per_place: int = 4,
        min_img_per_place: int = 4,
        random_sample_from_each_place: bool = True,
        transform: T.transforms.Compose = default_transform,
        root_dir: str = BASE_PATH,
        dataframes_dir: str = DF_PATH,
    ) -> Dataset:
        """
        Initialize the class for GSV-CITIES Dataset.

        Parameters:
            cities (list): List of cities to load images from.
            img_per_place (int): Number of images to load per place.
            min_img_per_place (int): Minimum number of images per place.
            random_sample_from_each_place (bool): Flag indicating whether to randomly sample images from each place.
            transform (torchvision.transforms.Compose): Transformation to apply to the images.
            base_path (str): Base path to the dataset folder.

        Raises:
            AssertionError: If img_per_place is greater than min_img_per_place.

        Returns:
            None
        """
        super(GSVCitiesDataset, self).__init__(
            root_dir=root_dir, dataframes_dir=dataframes_dir, transform=transform
        )
        self.cities = cities

        assert (
            img_per_place <= min_img_per_place
        ), f"img_per_place should be less than {min_img_per_place}"

        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place

        self.dfs = self.get_dataframes_list()
        self.dataframe = self.__getdataframes()

        # place_ids is a list, where each element is a UNIQUE place_id of a city
        self.places_ids = pd.unique(self.dataframe.index)
        # self.total_nb_images already inherited
        self.total_nb_images = len(self.dataframe)

    def __getdataframes(self) -> pd.DataFrame:
        """
        Return one dataframe containing all information about the images from all cities.

        This requires DataFrame files to be in a folder named Dataframes, containing a DataFrame
        for each city in the list of cities passed to the constructor.

        The returned dataframe is a concatenation of the dataframes for each city,
        with all duplicate place_ids removed.
        """
        # read the first city dataframe
        df = pd.read_csv(self.dataframes_dir +
                         f"\\{self.cities[0].lower()}.csv")
        df = df.sample(frac=1)  # shuffle the city dataframe

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(self.dataframes_dir +
                                 f"\\{self.cities[i].lower()}.csv")

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            prefix = i
            tmp_df["place_id"] = tmp_df["place_id"] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe

            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[
            df.groupby("place_id")["place_id"].transform("size")
            >= self.min_img_per_place
        ]
        return res.set_index("place_id")

    def get_df(self) -> pd.DataFrame:
        return self.dataframe

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a place and its corresponding images from the dataset based on the index.

        Parameters:
            index (int): The index of the place to retrieve.

        Returns:
            torch.Tensor: A tensor containing K images of the place.
            torch.Tensor: A tensor containing the place ID repeated K times.
        """
        place_id = self.places_ids[index]

        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]

        city = place["class_name"].iloc[0]
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place[: self.img_per_place]

        imgs = []
        for i, row in place.iterrows():
            img_path = self.get_img_name(row)
            img = self.image_loader(os.path.join(
                self.root_dir, city, img_path))

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)

        # NOTE: contrary to image classification where __getitem__ returns only one image
        # in GSVCities, we return a place, which is a Tensor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self) -> int:
        """Denotes the total number of places (not images)"""
        return len(self.places_ids)

    @staticmethod
    def image_loader(path) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def get_img_name(row) -> str:
        # given a row from the dataframe
        # return the corresponding image name

        img_name = row["filename"]

        return img_name


if __name__ == "__main__":
    dataset = GSVBaseDataset()
    dataset.show_random_images(n=2)
