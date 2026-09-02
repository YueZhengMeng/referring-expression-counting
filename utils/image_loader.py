import os

from torch.utils.data import Dataset, DataLoader

from groundingdino.util.base_api import load_image, preprocess_caption
from utils.processor import DataProcessor


def collate_fn(batch):
    images, labels, shapes, img_ids = zip(*batch)

    # Keep variable-sized tensors unpadded. GroundingDINO will create a
    # NestedTensor and the corresponding padding mask at model input time.
    images = list(images)

    # tuple to list
    labels = list(labels)
    shapes = list(shapes)
    img_ids = list(img_ids)

    # list of (3,h,w) tensors, list (), list ((h,w)), list ()
    return images, labels, shapes, img_ids


def get_loader(processor: DataProcessor, split, batch_size):
    # 获取划分的子集
    split_set = Rec8KDataset(processor, split)
    # 训练集打乱(shuffle=True)，使得每一个epoch中的数据顺序不同，防止过拟合
    # 验证集与测试集不打乱(shuffle=False)，使得评测结果稳定
    shuffle = True if split == 'train' else False
    split_loader = DataLoader(split_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return split_loader


class Rec8KDataset(Dataset):
    def __init__(self, processor: DataProcessor, split):
        self.processor = processor
        self.split = split

        # list of (img_id, cap)
        split_set_tuples = processor.get_img_ids_for_split(split)

        # {image_id: [caption_1, caption_2, ...], ...}
        split_dict = {}
        for img_id, cap in split_set_tuples:
            if img_id in split_dict:
                split_dict[img_id].append(cap)
            else:
                split_dict[img_id] = [cap]

        self.img_ids = list(split_dict.keys())
        self.labels = [list(split_dict[img_id]) for img_id in self.img_ids]  # list of list of caps

        # 保留原始caption用于精确查找原始annotation
        self.img_cap_tuples = []
        for i, (img_id, caps) in enumerate(zip(self.img_ids, self.labels)):
            img_cap_tuple = [(img_id, cap) for cap in caps]
            self.img_cap_tuples.append(img_cap_tuple)
            for j, cap in enumerate(caps):
                text_prompt = processor.get_prompt_for_image((img_id, cap))[0]
                # 预处理后的caption用于输入模型，确保结尾是句号
                self.labels[i][j] = preprocess_caption(caption=text_prompt)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = self.processor.get_image_path()
        img_file = os.path.join(img_path, self.img_ids[idx])

        # 加载原图，缩放，转tensor，归一化，转RGB
        # 缩放规则：最短边缩放到800，长边不超过1333
        # 长边过长则等比缩放到1333
        # image_source：np.array, (h, w, 3)
        # image：torch.tensor, (3, h, w)
        image_source, image = load_image(img_file)
        h, w, _ = image_source.shape

        # list of caps for same image
        label = self.labels[idx]

        # list of tuples (img_id, cap) for same image
        img_cap_tuple = self.img_cap_tuples[idx]

        #  预处理后的image tensor, 预处理后的label list, 原始image的尺寸, 原始的img_cap_tuple
        return image, label, (h, w), img_cap_tuple
