import unittest

import numpy as np
import torch

from projects.anomaly import data


def _fake_patches(n: int = 4, size: int = 96):
    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(n)]
    labels = [i % 2 for i in range(n)]
    return patches, labels


class TestTissueMask(unittest.TestCase):

    def test_when_white_background_then_mostly_false(self):
        # Arrange
        white = np.full((10, 10, 3), 255, dtype=np.uint8)

        # Act
        mask = data.tissue_mask(white)

        # Assert
        self.assertFalse(mask.any())


class TestPointInPolygons(unittest.TestCase):

    def test_when_point_inside_square_then_true(self):
        # Arrange
        square = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)]

        # Act
        actual = data.point_in_polygons(5, 5, square)

        # Assert
        self.assertTrue(actual)

    def test_when_point_outside_square_then_false(self):
        # Arrange
        square = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)]

        # Act
        actual = data.point_in_polygons(50, 50, square)

        # Assert
        self.assertFalse(actual)


class TestPatchDataset(unittest.TestCase):

    def test_when_indexed_then_returns_tensor_and_int_label(self):
        # Arrange
        patches, labels = _fake_patches()
        dataset = data.PatchDataset(patches, labels)
        expected_len = len(patches)

        # Act
        image, label = dataset[0]
        actual_len = len(dataset)

        # Assert
        self.assertEqual(actual_len, expected_len)
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 96, 96))
        self.assertIsInstance(label, int)

    def test_when_train_transform_then_returns_normalized_tensor(self):
        # Arrange
        patches, labels = _fake_patches()
        dataset = data.PatchDataset(patches, labels, transform=data.default_transform(train=True))

        # Act
        image, _ = dataset[0]

        # Assert
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 96, 96))

    def test_when_lengths_mismatch_then_raises_value_error(self):
        # Arrange
        patches, _ = _fake_patches(n=4)
        wrong_labels = [0, 1]

        # Act & Assert
        with self.assertRaises(ValueError):
            data.PatchDataset(patches, wrong_labels)


if __name__ == '__main__':
    unittest.main()
