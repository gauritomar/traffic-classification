from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image, ImageOps
import random
import os

class TrafficDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentation=True):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(['empty', 'low', 'medium', 'high', 'traffic-jam'])
        self.class_to_idx = {
            "empty": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "traffic-jam": 4,
        }
        self.img_paths, self.labels = self._load_data()

        if augmentation:
            self._augment_data()

    def _load_data(self):
        img_paths = []
        labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_path.endswith(".jpg"):
                    img_paths.append(img_path)
                    labels.append(self.class_to_idx[cls_name])
        return img_paths, labels

    def _augment_data(self):
        augmented_img_paths = []
        augmented_labels = []
        for img_path, label in zip(self.img_paths, self.labels):
            image = Image.open(img_path).convert("RGB")
            for _ in range(4):
                augmented_image = self.apply_augmentation(image)
                augmented_img_paths.append(img_path)
                augmented_labels.append(label)

        self.img_paths.extend(augmented_img_paths)
        self.labels.extend(augmented_labels)

    def apply_augmentation(self, image):
        augmented_image = image.copy()

        if random.random() < 0.5:
            augmented_image = ImageOps.mirror(augmented_image)

        width, height = augmented_image.size
        max_offset_x = int(width * 0.4)
        max_offset_y = int(height * 0.4)

        left = random.randint(0, max_offset_x)
        top = random.randint(0, max_offset_y)
        right = random.randint(width - max_offset_x, width)
        bottom = random.randint(height - max_offset_y, height)

        augmented_image = augmented_image.crop((left, top, right, bottom))

        jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        augmented_image = jitter(augmented_image)

        return augmented_image

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.img_paths)


