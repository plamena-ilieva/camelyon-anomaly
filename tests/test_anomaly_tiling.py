"""Тестове за извличането на патчове с подменен (mock) openslide.

`openslide` е тежка нативна зависимост, която иска реален whole-slide файл.
Тук я заместваме с фалшив слайд, за да тестваме логиката на tiling-а
(tissue mask, етикетиране по полигони, лимити) без данни.
"""
import os
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import Image


class _FakeSlide:
    level_downsamples = [1.0, 2.0, 4.0]
    level_dimensions = [(400, 400), (200, 200), (100, 100)]

    def __init__(self, path):
        pass

    def read_region(self, location, level, size):
        # хомогенно "тъканно" парче (наситен цвят -> минава tissue mask)
        arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3] = 180, 100, 150, 255
        return Image.fromarray(arr, 'RGBA')

    def close(self):
        pass


_fake = types.ModuleType('openslide')
_fake.OpenSlide = lambda path: _FakeSlide(path)
sys.modules['openslide'] = _fake

from projects.anomaly import data  # noqa: E402

_SQUARE_XML = ('<ASAP_Annotations><Annotations><Annotation><Coordinates>'
               '<Coordinate Order="0" X="0" Y="0"/><Coordinate Order="1" X="400" Y="0"/>'
               '<Coordinate Order="2" X="400" Y="400"/><Coordinate Order="3" X="0" Y="400"/>'
               '</Coordinates></Annotation></Annotations></ASAP_Annotations>')


def _write_square_xml():
    f = tempfile.NamedTemporaryFile('w', suffix='.xml', delete=False)
    f.write(_SQUARE_XML)
    f.close()
    return f.name


class TestExtractTilesBalanced(unittest.TestCase):

    def test_when_no_annotation_then_only_normal_tiles(self):
        # Act
        tiles = data.extract_tiles_balanced('slide.tif',
                                            tile_size=96,
                                            level=0,
                                            annotation_path=None,
                                            n_tumor=0,
                                            n_normal=5)
        # Assert
        self.assertEqual(len(tiles), 5)
        self.assertTrue(all(label == data.NORMAL for _, label in tiles))

    def test_when_polygon_covers_slide_then_tumor_tiles(self):
        # Arrange
        path = _write_square_xml()
        # Act
        tiles = data.extract_tiles_balanced('slide.tif',
                                            tile_size=96,
                                            level=0,
                                            annotation_path=path,
                                            n_tumor=5,
                                            n_normal=0)
        os.unlink(path)
        # Assert
        self.assertEqual(len(tiles), 5)
        self.assertTrue(all(label == data.TUMOR for _, label in tiles))


class TestExtractTiles(unittest.TestCase):

    def test_when_max_tiles_set_then_stops_at_limit(self):
        # Arrange
        path = _write_square_xml()
        # Act
        tiles = data.extract_tiles('slide.tif',
                                   tile_size=96,
                                   level=0,
                                   annotation_path=path,
                                   max_tiles=5)
        os.unlink(path)
        # Assert — цялата маска е тумор, лимитът е спазен
        self.assertEqual(len(tiles), 5)
        self.assertTrue(all(label == data.TUMOR for _, label in tiles))


if __name__ == '__main__':
    unittest.main()
