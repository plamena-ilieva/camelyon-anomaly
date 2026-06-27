import unittest

import torch

from projects.anomaly import models


class TestSimpleCNN(unittest.TestCase):

    def test_when_forward_then_returns_logits_per_class(self):
        # Arrange
        batch_size, num_classes = 2, 2
        model = models.SimpleCNN(num_classes=num_classes)
        x = torch.randn(batch_size, 3, 96, 96)
        expected_shape = (batch_size, num_classes)

        # Act
        actual = model(x)

        # Assert
        self.assertEqual(actual.shape, expected_shape)


class TestVGG(unittest.TestCase):

    def test_when_vgg16_forward_then_returns_logits_per_class(self):
        # Arrange
        batch_size, num_classes = 2, 2
        model = models.VGG(config='VGG16', num_classes=num_classes)
        x = torch.randn(batch_size, 3, 96, 96)
        expected_shape = (batch_size, num_classes)

        # Act
        actual = model(x)

        # Assert
        self.assertEqual(actual.shape, expected_shape)

    def test_when_unknown_config_then_raises_value_error(self):
        # Arrange
        bad_config = 'VGG999'

        # Act & Assert
        with self.assertRaises(ValueError):
            models.VGG(config=bad_config)


class TestUNetAutoencoder(unittest.TestCase):

    def test_when_reconstructing_then_output_shape_equals_input(self):
        # Arrange
        model = models.UNetAutoencoder(in_channels=3, out_channels=3, features=(16, 32))
        x = torch.randn(2, 3, 64, 64)

        # Act
        actual = model(x)

        # Assert
        self.assertEqual(actual.shape, x.shape)

    def test_when_segmentation_mode_then_single_output_channel(self):
        # Arrange
        model = models.UNetAutoencoder(in_channels=3, out_channels=1, features=(16, 32))
        x = torch.randn(2, 3, 64, 64)
        expected_shape = (2, 1, 64, 64)

        # Act
        actual = model(x)

        # Assert
        self.assertEqual(actual.shape, expected_shape)


class TestConvAutoencoder(unittest.TestCase):

    def test_when_reconstructing_then_output_shape_equals_input(self):
        # Arrange
        model = models.ConvAutoencoder(in_channels=3)
        x = torch.randn(2, 3, 96, 96)

        # Act
        actual = model(x)

        # Assert
        self.assertEqual(actual.shape, x.shape)

    def test_when_forward_then_output_in_unit_range(self):
        # Arrange — Sigmoid изход => стойности в [0, 1]
        model = models.ConvAutoencoder(in_channels=3)
        x = torch.randn(2, 3, 96, 96)

        # Act
        out = model(x)

        # Assert
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)


if __name__ == '__main__':
    unittest.main()
