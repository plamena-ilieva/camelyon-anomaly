import os
import tempfile
import unittest

import matplotlib

matplotlib.use('Agg')

import numpy as np

from projects.anomaly import training_curves as tc


def _synthetic(n: int = 48):
    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(n)]
    labels = np.array([i % 2 for i in range(n)])
    groups = []
    for i in range(n):
        if labels[i] == 0:
            groups.append('a' if (i // 2) % 2 == 0 else 'b')
        else:
            groups.append('c' if (i // 2) % 2 == 0 else 'd')
    return patches, labels, np.array(groups)


class TestLoadChunks(unittest.TestCase):

    def test_when_chunk_exists_then_loads_patches(self):
        # Arrange
        with tempfile.TemporaryDirectory() as d:
            np.savez_compressed(os.path.join(d, 's1.npz'),
                                patches=np.zeros((3, 96, 96, 3), dtype=np.uint8),
                                labels=np.array([0, 1, 0]),
                                patient='patient_x')

            # Act
            patches, labels, groups = tc.load_chunks(d)

        # Assert
        self.assertEqual(len(patches), 3)
        self.assertEqual(len(labels), 3)
        self.assertTrue((groups == 'patient_x').all())


class TestTrainWithHistory(unittest.TestCase):

    def test_when_trained_then_history_and_plots(self):
        # Arrange
        patches, labels, groups = _synthetic()

        # Act
        history = tc.train_with_history(patches, labels, groups, num_epochs=1, subsample=0)

        # Assert
        for key in ('train_loss', 'val_loss', 'train_kappa', 'val_kappa'):
            self.assertEqual(len(history[key]), 1)

        with tempfile.TemporaryDirectory() as d:
            tc.plot_curves(history, d)
            self.assertTrue(os.path.exists(os.path.join(d, 'curve_loss.png')))
            self.assertTrue(os.path.exists(os.path.join(d, 'curve_kappa.png')))


if __name__ == '__main__':
    unittest.main()
