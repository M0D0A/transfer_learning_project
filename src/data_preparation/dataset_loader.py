import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from .preprocessing import get_transforms


class MyCustomDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.transforms = transforms
        self.path = path
        self.data = []
        self.data_len = 0
        self.count_to_categories = {}

        for file_path, dir_list, file_list in os.walk(self.path):
            if len(dir_list) == 0 and len(file_list) != 0:
                target = int(file_path.split("/")[-1])

                for file_name in file_list:
                    self.data.append((os.path.join(file_path, file_name), target))

                self.data_len += len(file_list)
                self.count_to_categories[target] = len(file_list)

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        img_path, target = self.data[index]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


def create_dataset(path, split=None) -> MyCustomDataset:
    dataset = MyCustomDataset(
        path=path,
        transforms=get_transforms()
    )
    if split is not None:
        return random_split(dataset, split)
    return dataset


def create_loader(dataset, batch_size, shuffle=True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
