import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from projects.anomaly import data, models, train


def _loader():
    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(8)]
    labels = [0, 1] * 4
    dataset = data.PatchDataset(patches, labels)
    return DataLoader(dataset, batch_size=4)


class TestFit(unittest.TestCase):

    def test_when_fit_with_val_then_history_tracks_best_epoch(self):
        # Arrange
        loader = _loader()
        model = models.SimpleCNN(num_classes=2)
        num_epochs = 2

        # Act
        history = train.fit(model,
                            loader,
                            loader,
                            num_epochs=num_epochs,
                            lr=1e-3,
                            device=torch.device('cpu'))

        # Assert
        self.assertEqual(len(history['train_loss']), num_epochs)
        self.assertIn('best_epoch', history)
        self.assertIn('best_val_cohen_kappa', history)


class TestEvaluate(unittest.TestCase):

    def test_when_evaluate_then_returns_accuracy_and_kappa(self):
        # Arrange
        loader = _loader()
        model = models.SimpleCNN(num_classes=2)

        # Act
        scores = train.evaluate(model, loader, device=torch.device('cpu'))

        # Assert
        self.assertIn('accuracy', scores)
        self.assertIn('cohen_kappa', scores)


if __name__ == '__main__':
    unittest.main()
