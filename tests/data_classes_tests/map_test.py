"""Add a doc string to my files."""

import numpy as np
import pytest

from ekf_slam_3d.data_classes.map import (
    Feature,
    Map,
    make_box_map_planar,
    make_random_map_planar,
)


def test_map_creation() -> None:
    """Test that a map can be created."""
    # Arrange
    map_features = [Feature(id=1, x=1, y=1), Feature(id=2, x=2, y=2)]

    # Act
    map_object = Map(features=map_features)
    map_object.append_feature(Feature(id=2, x=2, y=2))
    map_object.append_feature(Feature(id=3, x=2, y=2))

    # Assert
    assert map_object.features == map_features


@pytest.mark.parametrize(("x", "y"), [(2, 3), (3, 4)])
def test_map_adjust(x: int, y: int) -> None:
    """Test that features on the map are adjustable."""
    # Arrange
    map_object = Map()
    map_object.append_feature(Feature(id=2, x=1, y=2))

    # Act
    feature = Feature(id=2, x=x, y=y)
    map_object.update_feature_location(feature)

    # Assert
    assert map_object.features[0].x == x
    assert map_object.features[0].y == y


def test_map_box() -> None:
    """Test that a box map is generated."""
    # Arrange
    num_features = 100

    # Act
    map_object = make_box_map_planar(num_features=num_features)

    # Assert
    assert len(map_object.features) == num_features


def test_map_random():
    """Test that a random map is generated."""
    # Arrange
    num_features = 100

    # Act
    map_object = make_random_map_planar(num_features=num_features)

    # Assert
    assert len(map_object.features) == num_features


def test_feature_as_vector():
    """Test that a feature can be created and converted to a vector."""
    # Arrange
    feature = Feature(id=1, x=2.0, y=3.0)

    # Act
    feature_vector = feature.as_vector()

    # Assert
    np.testing.assert_array_almost_equal(
        feature_vector, np.array([[2.0], [3.0], [0.0]])
    )
