from typing import Tuple, List, Dict

from torchvision.datasets import ImageFolder

import helper


class CustomImageFolder(ImageFolder):
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Override this method to load from setting file instead of scanning directory
        """
        classes = list(helper.classes_to_idx().keys())
        classes_to_idx = helper.classes_to_idx()
        return classes, classes_to_idx
