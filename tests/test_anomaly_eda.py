import unittest

import numpy as np

from projects.anomaly import eda


def _patch(r, g, b):
    return np.stack([np.full(
        (96, 96), r), np.full(
            (96, 96), g), np.full((96, 96), b)], axis=-1).astype(np.uint8)


class TestClassDistribution(unittest.TestCase):

    def test_when_counts_then_returns_per_class(self):
        # Arrange
        labels = [0, 0, 1, 0, 1]
        expected = {0: 3, 1: 2}

        # Act
        actual = eda.class_distribution(labels)

        # Assert
        self.assertEqual(actual, expected)


class TestPixelStatistics(unittest.TestCase):

    def test_when_channels_differ_then_means_differ(self):
        # Arrange — константни, но различни канали
        patches = [_patch(50, 100, 150) for _ in range(10)]
        expected = [50 / 255, 100 / 255, 150 / 255]

        # Act
        stats = eda.pixel_statistics(patches)

        # Assert — каналните средни трябва да са различни и верни
        for actual, exp in zip(stats['mean'], expected):
            self.assertAlmostEqual(actual, exp, places=4)


class TestSummarize(unittest.TestCase):

    def test_when_called_then_reports_counts_and_shape(self):
        # Arrange
        patches = [_patch(50, 100, 150), _patch(60, 110, 160)]
        labels = [0, 1]

        # Act
        report = eda.summarize(patches, labels)

        # Assert
        self.assertEqual(report['num_observations'], 2)
        self.assertEqual(report['patch_shape'], (96, 96, 3))
        self.assertEqual(report['class_distribution'], {0: 1, 1: 1})


if __name__ == '__main__':
    unittest.main()
