import os
import tempfile
import unittest

import matplotlib

matplotlib.use('Agg')

import numpy as np
import torch

from projects.anomaly import app, data, eda, metrics, models, train


class TestMetricsEdgeCases(unittest.TestCase):

    def test_when_empty_then_cohen_kappa_zero(self):
        # Arrange
        empty = torch.tensor([], dtype=torch.long)

        # Act
        actual = metrics.cohen_kappa(empty, empty, num_classes=2)

        # Assert
        self.assertEqual(actual, 0.0)

    def test_when_single_class_perfect_then_cohen_kappa_one(self):
        # Arrange — само един клас => очаквана съгласуваност 1 => знаменател 0
        y = torch.zeros(5, dtype=torch.long)

        # Act
        actual = metrics.cohen_kappa(y, y, num_classes=2)

        # Assert
        self.assertEqual(actual, 1.0)

    def test_when_empty_then_accuracy_zero(self):
        # Act
        actual = metrics.accuracy(torch.tensor([]), torch.tensor([]))

        # Assert
        self.assertEqual(actual, 0.0)


class TestReconstructionError(unittest.TestCase):

    def test_when_called_then_returns_one_score_per_image(self):
        # Arrange
        model = models.UNetAutoencoder(in_channels=3, out_channels=3, features=(8, 16))
        images = torch.rand(2, 3, 64, 64)

        # Act
        errors = train.reconstruction_error(model, images, device=torch.device('cpu'))

        # Assert
        self.assertEqual(errors.shape, (2, ))


class TestUNetOddInput(unittest.TestCase):

    def test_when_non_power_of_two_input_then_output_matches(self):
        # Arrange — нечетен размер задейства интерполацията при skip връзките
        model = models.UNetAutoencoder(out_channels=3, features=(8, 16))
        x = torch.randn(1, 3, 70, 70)

        # Act
        out = model(x)

        # Assert
        self.assertEqual(out.shape, x.shape)


class TestEdaPlots(unittest.TestCase):

    def test_when_plotting_then_files_are_created(self):
        # Arrange
        rng = np.random.default_rng(0)
        patches = [rng.integers(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(6)]
        labels = [0, 1, 0, 1, 0, 1]

        # Act & Assert
        with tempfile.TemporaryDirectory() as d:
            dist = os.path.join(d, 'dist.png')
            grid = os.path.join(d, 'grid.png')
            eda.plot_class_distribution(labels, dist)
            eda.plot_sample_grid(patches, labels, grid, nrows=2, ncols=3)
            self.assertTrue(os.path.exists(dist))
            self.assertTrue(os.path.exists(grid))


class TestReadAnnotations(unittest.TestCase):

    def test_when_xml_has_polygon_then_parses_coordinates(self):
        # Arrange
        xml = ('<ASAP_Annotations><Annotations><Annotation><Coordinates>'
               '<Coordinate Order="0" X="1.0" Y="2.0"/>'
               '<Coordinate Order="1" X="3.0" Y="4.0"/>'
               '<Coordinate Order="2" X="5.0" Y="6.0"/>'
               '</Coordinates></Annotation></Annotations></ASAP_Annotations>')
        with tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False) as f:
            f.write(xml)
            path = f.name

        # Act
        polygons = data.read_annotations(path)
        os.unlink(path)

        # Assert
        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].shape, (3, 2))


class TestPatchDatasetFromDirectory(unittest.TestCase):

    def test_when_directory_has_classes_then_loads_all(self):
        # Arrange
        from PIL import Image
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as root:
            for cls in ('normal', 'tumor'):
                os.makedirs(os.path.join(root, cls))
                for i in range(2):
                    arr = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(os.path.join(root, cls, f'{i}.png'))

            # Act
            dataset = data.PatchDataset.from_directory(root)

            # Assert
            self.assertEqual(len(dataset), 4)


class TestLoadModelRawStateDict(unittest.TestCase):

    def test_when_plain_state_dict_then_uses_arch_hint(self):
        # Arrange — суров state_dict (без 'arch') => ползва се arch_hint
        import pathlib as pl
        path = pl.Path('tmp_raw_model.pt')
        torch.save(models.SimpleCNN().state_dict(), path)

        # Act
        model, arch = app._load_model(str(path), arch_hint='SimpleCNN')
        path.unlink()

        # Assert
        self.assertEqual(arch, 'SimpleCNN')
        self.assertIsInstance(model, models.SimpleCNN)


if __name__ == '__main__':
    unittest.main()
