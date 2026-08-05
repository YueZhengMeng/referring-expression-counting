import json
from pathlib import Path


class DataProcessor:
    def __init__(self, image_path, anno_file_path='anno/annotations.json',
                 split_file_path='anno/splits.json'):
        self.image_path = Path(image_path)
        self.anno_file = Path(anno_file_path)
        self.split_file = Path(split_file_path)
        if not self.image_path.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not self.anno_file.is_file():
            raise FileNotFoundError(f"Annotation file not found: {self.anno_file}")
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        self.annotations = self.read_annotations()
        self.splits = self.read_splits()

        print(f"annotation file: {self.anno_file}\nsplit file: {self.split_file}")
        print(f"train: {len(self.splits['train'])}\nval: {len(self.splits['val'])}\ntest: {len(self.splits['test'])}")

    def read_annotations(self):
        annotations = {}
        with self.anno_file.open(encoding='utf-8') as f:
            anno = json.load(f)
            for image_id, captions in anno.items():
                for caption, items in captions.items():
                    annotations[(image_id, caption)] = items
        return annotations

    def read_splits(self):
        with self.split_file.open(encoding='utf-8') as f:
            splits = json.load(f)
            return {key: [tuple(x) for x in item] for key, item in splits.items()}

    def get_image_path(self):
        return str(self.image_path)

    def get_anno_for_tuple(self, image_id, caption):
        return self.annotations[(image_id, caption)]

    def get_class_name(self, image_id, caption):
        return self.annotations[(image_id, caption)]['class']

    def get_attr_name(self, image_id, caption):
        return self.annotations[(image_id, caption)]['attribute']

    def get_type_name(self, image_id, caption):
        return self.annotations[(image_id, caption)]['type']

    def get_split_type(self, image_id, caption):
        for split_type, pairs in self.splits.items():
            if (image_id, caption) in pairs:
                return split_type
        return None

    def get_prompt_for_image(self, image_id_caption: tuple):
        if not isinstance(image_id_caption, tuple) or len(image_id_caption) != 2:
            raise TypeError('input must be a tuple of (image_id, caption)')
        _, caption = image_id_caption
        return [caption]

    def get_img_ids_for_split(self, split):
        if split not in self.splits:
            raise KeyError(f"Unknown split: {split}")
        return self.splits[split]
