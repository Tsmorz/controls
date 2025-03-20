"""Add a doc string to my files."""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from config.definitions import MAP_DIM, MAP_NUM_FEATURES


@dataclass
class Feature:
    """Dataclass to represent a detected feature."""

    id: int
    x: float
    y: float


class Map:
    """Dataclass to store features for a map."""

    def __init__(self, features=None):
        if features is None:
            features = []
        self.features: list[Feature] = features

    def append_feature(self, feature: Feature) -> None:
        """Append a feature to the map.

        :param feature: Feature to be appended
        :return: None
        """
        if self.feature_already_found(feature):
            logger.warning(f"Revisited landmark with I.D. {feature.id}.")
        else:
            self.features.append(feature)
            logger.info(f"Added landmark with I.D. {feature.id}.")

    def feature_already_found(self, new_feature: Feature) -> bool:
        """Check if the feature already exists.

        :param new_feature: Feature to check
        :return: True if the feature already exists, False otherwise
        """
        ids = [feature.id for feature in self.features]
        return new_feature.id in ids

    def update_feature_location(self, feature: Feature) -> None:
        """Update the location of a feature.

        :param feature: Feature to update
        :return: None
        """
        idx = [feature.id for feature in self.features].index(feature.id)
        self.features[idx].x = feature.x
        self.features[idx].y = feature.y

    def make_random_map_planar(
        self, num_features: int = MAP_NUM_FEATURES, dim: tuple[int, int] = MAP_DIM
    ) -> None:
        """Make a random map.

        :param num_features: Number of features to generate
        :param dim: Dimensions of the map
        :return: A map with random features
        """
        for num in range(num_features):
            map_feature = Feature(
                id=num,
                x=np.random.uniform(0, dim[0]),
                y=np.random.uniform(0, dim[1]),
            )
            self.append_feature(map_feature)

    def make_box_map_planar(
        self, num_features: int = MAP_NUM_FEATURES, dim: tuple[int, int] = MAP_DIM
    ) -> None:
        """Make a box map with random points.

        :param num_features: Number of features to generate
        :param dim: Dimensions of the map
        :return: A map with random features
        """
        num_features = 4 * round(num_features / 4)

        for num in range(0, int(num_features / 4)):
            x = np.random.uniform(0, dim[0])
            map_feature = Feature(id=num, x=x, y=0)
            self.append_feature(map_feature)
        for num in range(int(num_features / 4), int(num_features / 2)):
            y = np.random.uniform(0, dim[1])
            map_feature = Feature(id=num, x=dim[0], y=y)
            self.append_feature(map_feature)
        for num in range(int(num_features / 2), int(3 * num_features / 4)):
            x = np.random.uniform(0, dim[0])
            map_feature = Feature(id=num, x=x, y=dim[1])
            self.append_feature(map_feature)
        for num in range(int(3 * num_features / 4), num_features):
            y = np.random.uniform(0, dim[1])
            map_feature = Feature(id=num, x=0, y=y)
            self.append_feature(map_feature)
