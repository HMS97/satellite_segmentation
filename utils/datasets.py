"""PyTorch-compatible datasets.

Guaranteed to implement `__len__`, and `__getitem__`.

See: http://pytorch.org/docs/0.3.1/data.html
"""
from path import Path

import torch
from PIL import Image
import torch.utils.data
from torchvision.transforms import Compose, Normalize
from .transforms import ConvertImageMode, ImageToTensor

from .tiles import tiles_from_slippy_map, buffer_tile_image


mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Single Slippy Map directory structure
class SlippyMapTiles(torch.utils.data.Dataset):
    """Dataset for images stored in slippy map format.
    """

    def __init__(self, root, transform=None):
        super().__init__()

        self.tiles = []
        self.transform = transform

        self.tiles = [(tile, path) for tile, path in tiles_from_slippy_map(root)]
        self.tiles.sort(key=lambda tile: tile[0])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        tile, path = self.tiles[i]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, tile


# Multiple Slippy Map directories.
# Think: one with images, one with masks, one with rasterized traces.
class SlippyMapTilesConcatenation(torch.utils.data.Dataset):
    """Dataset to concate multiple input images stored in slippy map format.
    """

    def __init__(self, inputs, target, joint_transform=None,debug = False,test = False):
        super().__init__()

        # No transformations in the `SlippyMapTiles` instead joint transformations in getitem
        self.joint_transform = joint_transform
        self.test = test
        if debug == False:
            self.inputs =  Path(inputs).files()
            if self.test == False:
                self.target = Path(target).files()
        else:
            self.inputs =  Path(inputs).files()[:1000]
            if self.test == False:
                self.target =  Path(target).files()
        self.test_transform =Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])

    def __len__(self):
#         return len(self.target)
        return len(self.inputs)

    def __getitem__(self, i):
        # at this point all transformations are applied and we expect to work with raw tensors

        images = Image.open(self.inputs[i])
        if self.test == False:
            mask  = Image.open(self.target[i])
            if self.joint_transform is not None:
                images, mask = self.joint_transform(images, mask)

            return images, mask
        else:
            return self.test_transform(images)
# Todo: once we have the SlippyMapDataset this dataset should wrap
# it adding buffer and unbuffer glue on top of the raw tile dataset.
class BufferedSlippyMapDirectory(torch.utils.data.Dataset):
    """Dataset for buffered slippy map tiles with overlap.
    """

    def __init__(self, root, transform=None, size=512, overlap=32):
        """
        Args:
          root: the slippy map directory root with a `z/x/y.png` sub-structure.
          transform: the transformation to run on the buffered tile.
          size: the Slippy Map tile size in pixels
          overlap: the tile border to add on every side; in pixel.

        Note:
          The overlap must not span multiple tiles.

          Use `unbuffer` to get back the original tile.
        """

        super().__init__()

        assert overlap >= 0
        assert size >= 256

        self.transform = transform
        self.size = size
        self.overlap = overlap
        self.tiles = list(tiles_from_slippy_map(root))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        tile, path = self.tiles[i]
        image = buffer_tile_image(tile, self.tiles, overlap=self.overlap, tile_size=self.size)

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.IntTensor([tile.x, tile.y, tile.z])

    def unbuffer(self, probs):
        """Removes borders from segmentation probabilities added to the original tile image.

        Args:
          probs: the segmentation probability mask to remove buffered borders.

        Returns:
          The probability mask with the original tile's dimensions without added overlap borders.
        """

        o = self.overlap
        _, x, y = probs.shape

        return probs[:, o : x - o, o : y - o]
