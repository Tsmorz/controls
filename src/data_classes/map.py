"""Add a doc string to my files."""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from config.definitions import FIG_SIZE, MAP_DIM, MAP_NUM_FEATURES


@dataclass
class Feature:
    """Dataclass to represent a detected feature."""

    id: int
    x: float
    y: float
    z: float = 0.0


@dataclass
class Map:
    """Dataclass to store features for a map."""

    features: list[Feature] = field(default_factory=list)
    past_feature_ids: list[int] = field(default_factory=list)

    def append_feature(self, map_feature: Feature) -> None:
        """Append a feature to the map.

        :param map_feature: Feature to be appended
        :return: None
        """
        if self.feature_already_found(map_feature):
            logger.info(f"Revisited landmark with I.D. {map_feature.id}.")
        else:
            self.features.append(map_feature)
            self.past_feature_ids.append(map_feature.id)
            logger.info(f"Added landmark with I.D. {map_feature.id}.")

    def feature_already_found(self, new_feature: Feature) -> bool:
        """Check if the feature already exists.

        :param new_feature: Feature to check
        :return: True if the feature already exists, False otherwise
        """
        return new_feature.id in self.past_feature_ids

    def update_feature_location(self, feature: Feature) -> None:
        """Update the location of a feature.

        :param feature: Feature to update
        :return: None
        """
        idx = self.past_feature_ids.index(feature.id)
        self.features[idx].x = feature.x
        self.features[idx].y = feature.y


def make_random_map_planar(
    num_features: int = MAP_NUM_FEATURES, dim: tuple[int, int] = MAP_DIM
) -> Map:
    """Make a random map.

    :param num_features: Number of features to generate
    :param dim: Dimensions of the map
    :return: A map with random features
    """
    random_map = Map()
    for num in range(num_features):
        map_feature = Feature(
            id=num,
            x=np.random.uniform(0, dim[0]),
            y=np.random.uniform(0, dim[1]),
        )
        random_map.append_feature(map_feature)
    return random_map


def make_box_map_planar(
    num_features: int = MAP_NUM_FEATURES, dim: tuple[int, int] = MAP_DIM
) -> Map:
    """Make a box map with random points.

    :param num_features: Number of features to generate
    :param dim: Dimensions of the map
    :return: A map with random features
    """
    num_features = 4 * round(num_features / 4)
    box_map = Map()
    for num in range(0, int(num_features / 4)):
        x = np.random.uniform(0, dim[0])
        map_feature = Feature(id=num, x=x, y=0)
        box_map.append_feature(map_feature)
    for num in range(int(num_features / 4), int(num_features / 2)):
        y = np.random.uniform(0, dim[1])
        map_feature = Feature(id=num, x=dim[0], y=y)
        box_map.append_feature(map_feature)
    for num in range(int(num_features / 2), int(3 * num_features / 4)):
        x = np.random.uniform(0, dim[0])
        map_feature = Feature(id=num, x=x, y=dim[1])
        box_map.append_feature(map_feature)
    for num in range(int(3 * num_features / 4), num_features):
        y = np.random.uniform(0, dim[1])
        map_feature = Feature(id=num, x=0, y=y)
        box_map.append_feature(map_feature)
    return box_map


if __name__ == "__main__":  # pragma: no cover
    map_planar = make_random_map_planar()
    logger.info(f"Random map has {len(map_planar.past_feature_ids)} features.")

    map_box = make_box_map_planar()
    logger.info(f"Box map has {len(map_box.past_feature_ids)} features.")
    map_box.update_feature_location(feature=Feature(x=5, y=5, id=0))

    plt.figure(figsize=FIG_SIZE)
    plt.scatter(
        [feature.x for feature in map_box.features],
        [feature.y for feature in map_box.features],
        s=1,
    )
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Box Map")
    plt.show()
    plt.close()
