from PIL import Image
from torch.utils import data
from pathlib import Path


class ImageFolderDataset(data.Dataset):
    def __init__(
            self,
            root,
            transform=None,
            ext=('.png', '.jpg', '.jpeg', 'bmp'),
            recursive=False,
            return_filename=False
    ):
        if recursive:
            glob_pattern = '**/*'
        else:
            glob_pattern = '*'

        self.files = [f for f in Path(root).glob(glob_pattern) if f.is_file() and f.suffix.lower() in ext]
        self.transform = transform
        self.return_filename = return_filename
        self.mode = 'RGB'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        with open(filename, 'rb') as f:
            img = Image.open(f)
            img = img.convert(self.mode)
        if self.transform:
            img = self.transform(img)
        if self.return_filename:
            return img, str(filename)
        return img


class NoLabelWrapper(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][0]

    def __len__(self):
        return len(self.dataset)
