from typing import Optional, Any, List
import struct
import numpy as np

class MnistImage(object):
    def __init__(self, image: Any, label: Optional[int] = None):
        self.image = image
        self.label = label

    def to_tensor(self) -> Any:
        return (self.image.reshape(1, 1, 28, 28) / 255.0).astype(np.float32)


class MnistDataset(object):
    def __init__(self, dataset_path: str):
        self.images = self._load_images(dataset_path)

    def _load_images(self, dataset_path: str) -> List[MnistImage]:
        images = []
        with open(dataset_path, "rb") as dataset_file:
            _, num, rows, cols = struct.unpack(">IIII", dataset_file.read(16))
            images_array = np.fromfile(dataset_file, dtype=np.uint8).reshape(num, rows * cols)
            for image in images_array:
                images.append(MnistImage(image))

        return images

    def load_labels(self, labels_path: str):
        with open(labels_path, "rb") as labels_file:
            _, _ = struct.unpack(">II", labels_file.read(8))
            labels = np.fromfile(labels_file, dtype=np.uint8)
            for i, label in enumerate(labels):
                self.images[i].label = label
