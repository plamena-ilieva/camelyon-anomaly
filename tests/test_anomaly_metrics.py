import unittest

import torch

from projects.anomaly import metrics


class TestConfusionMatrix(unittest.TestCase):

    def test_when_perfect_predictions_then_diagonal_matrix(self):
        # Arrange
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0, 0, 1, 1])
        expected = torch.tensor([[2, 0], [0, 2]])

        # Act
        actual = metrics.confusion_matrix(y_true, y_pred, num_classes=2)

        # Assert
        self.assertTrue(torch.equal(actual, expected))


class TestCohenKappa(unittest.TestCase):

    def test_when_perfect_agreement_then_kappa_is_one(self):
        # Arrange
        y_true = torch.tensor([0, 1, 0, 1, 1])
        y_pred = torch.tensor([0, 1, 0, 1, 1])
        expected = 1.0

        # Act
        actual = metrics.cohen_kappa(y_true, y_pred, num_classes=2)

        # Assert
        self.assertAlmostEqual(actual, expected, places=5)

    def test_when_predictions_all_wrong_then_kappa_is_negative(self):
        # Arrange
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([1, 1, 0, 0])

        # Act
        actual = metrics.cohen_kappa(y_true, y_pred, num_classes=2)

        # Assert
        self.assertLess(actual, 0.0)

    def test_when_matches_sklearn_then_close(self):
        # Arrange
        y_true = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = torch.tensor([0, 1, 0, 0, 1, 1, 1, 1])
        from sklearn.metrics import cohen_kappa_score
        expected = cohen_kappa_score(y_true.numpy(), y_pred.numpy())

        # Act
        actual = metrics.cohen_kappa(y_true, y_pred, num_classes=2)

        # Assert
        self.assertAlmostEqual(actual, float(expected), places=5)


class TestAccuracy(unittest.TestCase):

    def test_when_half_correct_then_returns_half(self):
        # Arrange
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0, 1, 0, 1])
        expected = 0.5

        # Act
        actual = metrics.accuracy(y_true, y_pred)

        # Assert
        self.assertAlmostEqual(actual, expected, places=5)


class TestF1Score(unittest.TestCase):

    def test_when_perfect_then_f1_is_one(self):
        # Arrange
        y_true = torch.tensor([0, 1, 1, 0])
        y_pred = torch.tensor([0, 1, 1, 0])

        # Act
        actual = metrics.f1_score(y_true, y_pred)

        # Assert
        self.assertAlmostEqual(actual, 1.0, places=5)

    def test_when_matches_sklearn_then_close(self):
        # Arrange
        y_true = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = torch.tensor([0, 1, 0, 0, 1, 1, 1, 1])
        from sklearn.metrics import f1_score as sk_f1
        expected = sk_f1(y_true.numpy(), y_pred.numpy())

        # Act
        actual = metrics.f1_score(y_true, y_pred)

        # Assert
        self.assertAlmostEqual(actual, float(expected), places=5)


if __name__ == '__main__':
    unittest.main()
