import pathlib as pl
import unittest

import numpy as np
import torch

from projects.anomaly import app, models


class TestPredictPatch(unittest.TestCase):

    def test_when_called_then_returns_label_and_probabilities(self):
        # Arrange
        model = models.SimpleCNN(num_classes=2)
        patch = np.random.default_rng(0).integers(0, 256, (96, 96, 3), dtype=np.uint8)

        # Act
        result = app.predict_patch(model, patch)

        # Assert
        self.assertIn(result['label'], (0, 1))
        self.assertAlmostEqual(result['prob_normal'] + result['prob_tumor'], 1.0, places=4)
        self.assertGreaterEqual(result['prob_tumor'], 0.0)
        self.assertLessEqual(result['prob_tumor'], 1.0)


class TestLoadModel(unittest.TestCase):

    def test_when_checkpoint_has_arch_then_loads_that_architecture(self):
        # Arrange
        path = pl.Path('tmp_model.pt')
        torch.save({'arch': 'VGG11', 'state_dict': models.VGG(config='VGG11').state_dict()}, path)

        # Act
        model, arch = app._load_model(str(path))

        # Assert
        self.assertEqual(arch, 'VGG11')
        self.assertIsInstance(model, models.VGG)

        # Clean
        path.unlink()

    def test_when_path_missing_then_returns_none(self):
        # Act
        model, arch = app._load_model('does_not_exist.pt')

        # Assert
        self.assertIsNone(model)
        self.assertIsNone(arch)


if __name__ == '__main__':
    unittest.main()
