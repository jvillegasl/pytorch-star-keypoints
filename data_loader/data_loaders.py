from pathlib import Path
from base import BaseDataLoader
from data_loader.coco import CocoDetection, make_coco_transforms
from utils import collate_fn


class StarsKptsDataLoader(BaseDataLoader):
    def __init__(
            self,
            data_dir,
            batch_size,
            shuffle=True,
            validation_split=0.1,
            num_workers=1,
            training=True
    ):
        mode = 'train' if training else 'test'

        trsfm = make_coco_transforms(mode)
        self.data_dir = Path(data_dir)
        assert self.data_dir, f'provided COCO path {self.data_dir} does not exist'

        img_folder = self.data_dir / mode
        ann_file = self.data_dir / (mode + '.json')

        self.dataset = CocoDetection(
            img_folder, ann_file, transforms=trsfm, return_masks=False)

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, collate_fn=collate_fn)
