import unittest

import numpy as np

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


if __name__ == '__main__':
    unittest.main()
