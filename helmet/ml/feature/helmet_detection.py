import torch
from torchvision import datasets
from pycocotools.coco import COCO
from helmet.constants import ANNOTATIONS_COCO_JSON_FILE
from helmet.exception import HelmetException
import cv2
import os
import sys
import copy


class HelmetDetection(datasets.VisionDataset):

    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        # the 3 transform parameters are required for datasets.VisionDataset
        super().__init__(root, transforms, transform, target_transform)
        self.split = split #train, valid, test
        self.coco = COCO(os.path.join(root, split, ANNOTATIONS_COCO_JSON_FILE)) # annotation stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
    
    def _load_image(self, id: int):
        try:
            path = self.coco.loadImgs(id)[0]['file_name']
            image = cv2.imread(os.path.join(self.root, self.split, path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            raise HelmetException(e, sys) from e

    def _load_target(self, id):
        try:
            return self.coco.loadAnns(self.coco.getAnnIds(id))
        except Exception as e:
            raise HelmetException(e, sys) from e
    
    def __getitem__(self, index):
        try:
            id = self.ids[index]
            image = self._load_image(id)
            target = self._load_target(id)
            target = copy.deepcopy(self._load_target(id))
            
            boxes = [t['bbox'] + [t['category_id']] for t in target]  # required annotation format for albumentations
            if self.transforms is not None:
                transformed = self.transforms(image=image, bboxes=boxes)
            
            image = transformed['image']
            boxes = transformed['bboxes']
            
            new_boxes = [] # convert from xywh to xyxy
            for box in boxes:
                xmin = box[0]
                xmax = xmin + box[2]
                ymin = box[1]
                ymax = ymin + box[3]
                new_boxes.append([xmin, ymin, xmax, ymax])
        
            boxes = torch.tensor(new_boxes, dtype=torch.float32)
            
            targ = {} # here is our transformed target
            targ['boxes'] = boxes
            targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
            targ['image_id'] = torch.tensor([t['image_id'] for t in target])
            targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
            targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
            return image.div(255), targ # scale images
        
        except Exception as e:
            raise HelmetException(e, sys) from e

    def __len__(self):
        try:
            return len(self.ids)

        except Exception as e:
            raise HelmetException(e, sys) from e
