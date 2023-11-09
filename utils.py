import cv2, shutil, yaml, requests, tempfile, torch
import kwcoco
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

class YOLO_dataset:
    def __init__(self, dataset_yaml:str, dataset_key:str) -> None:
        self.current = 0
        
        with open(dataset_yaml) as file:
            obj = yaml.safe_load(file)
            
            self.dataset_path = Path(dataset_yaml).parent

            self.img_data_dir = Path(self.dataset_path)/Path(obj[dataset_key])
            self.label_data_dir = self.img_data_dir.parent/"labels"

            self.class_num  = obj["nc"]
            self.class_name = obj["names"]

            if "nkpt" in obj:
                self.keypoint_num = obj["nkpt"]
            else:
                self.keypoint_num = None

            if "kpt_shape" in obj:
                self.kpt_shape = obj["kpt_shape"]
                self.keypoint_num = [obj["kpt_shape"][0] for _ in range(self.class_num)]
            else:
                self.kpt_shape = None

        self.img_paths   = sorted([str(path) for path in self.img_data_dir.glob("*") if path.suffix[1:] in ("jpg", "jpeg", "png")])
        self.label_paths = sorted([str(path) for path in self.label_data_dir.glob("*") if path.suffix[1:] in ("txt")])

    def __getitem__(self, key):
        data = {}

        data["img_path"] = Path(self.img_paths[key])
        data["label_path"] = Path(self.label_paths[key])
        if data["img_path"].stem != data["label_path"].stem:
            data["label_path"] = self.label_data_dir / (data["img_path"].stem + ".txt")

        img = cv2.imread(str(data["img_path"]))
        data["orig_img"] = img
        data["orig_shape"] = img.shape

        labels, bboxes, masks, keypoints = [], [], [], []
        with open(str(data["label_path"])) as f:
            annos = [l.split(" ") for l in f.read().splitlines()]
            for ann in annos:
                ann = list(map(float, ann))
                cls, center_x, center_y, width, height = ann[:5]

                w = width  * img.shape[1]
                h = height * img.shape[0]
                x1 = ((2 * center_x * img.shape[1]) - w)/2
                y1 = ((2 * center_y * img.shape[0]) - h)/2
                x2 = ((2 * center_x * img.shape[1]) + w)/2
                y2 = ((2 * center_y * img.shape[0]) + h)/2
                prob = 0
                bbox = [int(x1), int(y1), int(x2), int(y2), prob, cls]

                labels.append(str(self.class_name[int(cls)]))
                bboxes.append(bbox)

                if self.keypoint_num is not None:
                    kp = ann[5:5+(self.keypoint_num[int(cls)]*3)]
                    kp = [kp[idx:idx + 3] for idx in range(0, len(kp), 3)]
                    kp = [[kp[0]*img.shape[1], kp[1]*img.shape[0], kp[2]] for kp in kp]
                    if len(kp) > 0:
                        keypoints.append(kp)
                
                else:
                    seg = ann[5:]
                    seg = [seg[idx:idx + 2] for idx in range(0, len(seg), 2)]
                    seg = [[seg[0]*img.shape[1], seg[1]*img.shape[0]] for seg in seg]
                    if len(seg) > 0:
                        masks.append(seg)

        data["labels"]    = labels
        data["bboxes"]    = np.array(bboxes, dtype=float)
        data["keypoints"] = np.array(keypoints, dtype=float)
        data["masks"]     = self._get_mask_img(masks, img.shape)
        
        return data
        
    def __len__(self):
        return len(self.img_paths)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current == len(self):
            raise StopIteration()
        self.current += 1
        return self[self.current-1]
    
    def _get_mask_img(self, masks, img_shape) -> np.ndarray:
        mask_img = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)

        masks = [np.array(p).reshape((-1,2)).astype(np.int32) for p in masks]
        mask_img = cv2.fillPoly(mask_img, masks, color=1)
        print(mask_img.shape)

        mask_img = torch.from_numpy(mask_img)
        return mask_img

def show_img(img, title=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
    ax.set_title( title, fontsize=16, color='black')
    ax.axes.xaxis.set_visible(False) # X軸を非表示に
    ax.axes.yaxis.set_visible(False) # Y軸を非表示に
    ax.imshow(img)
    return fig, ax

def show_imgs(imgs_dict:dict, ncol=0, dpi=200, font_scale=0.3):
    font_size = int(plt.rcParams["font.size"]*font_scale)

    if ncol > 0:
        nrow = ((len(imgs_dict)-1)//ncol)+1
    else:
        nrow = 1
        ncol = len(imgs_dict)

    img_num = len(imgs_dict)
    fig = plt.figure(figsize=(float(img_num), float(img_num)), dpi=dpi)
    grid = ImageGrid(fig, 111, nrows_ncols=(nrow, ncol), axes_pad=0.2,)

    for i in range(nrow*ncol):
        grid[i].axis('off')
        if i < len(imgs_dict):
            img_key = list(imgs_dict.keys())[i]
            grid[i].imshow(imgs_dict[img_key])
            grid[i].set_title(img_key, fontsize=font_size, color='black', pad=int(font_size/2))
    
    plt.show(); plt.close()
    return None

def coco2yolo(
                output_dir:str,  
                train:kwcoco.CocoDataset = None, 
                val:kwcoco.CocoDataset   = None, 
                test:kwcoco.CocoDataset  = None,
                task:str           = "detection", # "detection" or "segmentation" or "keypoint"
                config_name:str    = "config.yaml",
                dataset_path:str   = None,
                cat_file_name:str  = "categories.txt",
                exist_ok:bool      = False,
            ):

    # trainがNoneの場合はエラー
    # さらに、train, val, testが全てNoneの場合はエラー
    if train is None:
        raise ValueError("train is None")
    elif train is None and val is None and test is None:
        raise ValueError("train, val, test are all None")
    
    # output_dirがディレクトリでない場合はエラー
    if Path(output_dir).is_file():
        raise ValueError("output_dir is not directory")
    
    # output_dirが存在し、exist_okがTrueの場合は、削除して新規作成
    # output_dirが存在しなければ作成
    if Path(output_dir).exists() and exist_ok:
        shutil.rmtree(Path(output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=exist_ok)

    config_path = Path(output_dir)/config_name
    cat_file_path = Path(output_dir)/cat_file_name

    # trainをもとにしてconfigファイルを作成する
    # CatIdが1から始まるようにする
    coco_to_yolo_catId = {}
    coutner = 1
    for catId in list(train.cats.keys()):
        if task == "detection" or task == "segmentation":
            coco_to_yolo_catId[catId] = coutner
            coutner += 1
        elif task == "keypoint":
            try:
                if len(train.cats[catId]['keypoints']) > 0:
                    coco_to_yolo_catId[catId] = coutner
                    coutner += 1
            except:
                continue

    # カテゴリーを書き出す
    with open(str(cat_file_path), mode = "w") as f:
        for coco_catId in list(train.cats.keys()):
            try:
                f.write(f"{coco_to_yolo_catId[coco_catId]}: {train.cats[coco_catId]['name']}\n")
            except:
                if task == "detection" or task == "segmentation":
                    f.write(f"{coco_to_yolo_catId[coco_catId]}:\n")
                elif task == "keypoint":
                    continue

    # YOLO学習用configを書き出す
    if not config_path.exists():
        with open(str(config_path), mode = "w") as f:
            if dataset_path is None or dataset_path == "":
                f.write(f"path:  {Path(output_dir).name}\n")
            else:
                f.write(f"path:  {str(dataset_path)}\n")

            f.write(f"train: train/images\n")
            f.write(f"val:   valid/images\n")
            f.write(f"test:  test/images\n")
            f.write(f"\n")
            f.write(f"nc: {len(coco_to_yolo_catId)+1}\n")
            f.write(f"\n")

            if task == "keypoint":
                f.write(f"nkpt:\n")
                f.write(f"  0: 0\n")
                for coco_catId in list(train.cats.keys()):
                    try:
                        f.write(f"  {coco_to_yolo_catId[coco_catId]}: {len(train.cats[coco_catId]['keypoints'])}\n")
                    except:
                        continue
                f.write(f"\n")

            f.write(f"names:\n")
            f.write(f"  0:\n")
            for coco_catId in list(train.cats.keys()):
                try:
                    f.write(f"  {coco_to_yolo_catId[coco_catId]}: {train.cats[coco_catId]['name']}\n")
                except:
                    if task == "detection" or task == "segmentation":
                        f.write(f"{coco_to_yolo_catId[coco_catId]}:\n")
                    elif task == "keypoint":
                        continue

    dataset_names = ["train", "valid", "test"]
    for coco, dataset_name in zip([train, val, test], dataset_names):
        if coco is None:
            continue

        # ディレクトリを作成する
        base_save_dir = Path(output_dir)/dataset_name
        save_images_dir = base_save_dir / Path("images")
        save_labels_dir = base_save_dir / Path("labels")
        [dr.mkdir(parents=True, exist_ok=exist_ok) for dr in [base_save_dir,save_images_dir,save_labels_dir]]

        gids = list(coco.imgs.keys())
        for gid in tqdm(gids, desc=f"coco2yolo:{dataset_name}"):

            # 画像ファイルをコピーする
            img_src_path = Path(coco.get_image_fpath(gid))
            img_dst_path = save_images_dir/img_src_path.name
            shutil.copyfile(img_src_path, img_dst_path)

            # ラベルファイルを作成する
            img = coco.load_image(gid)
            label_path = f'{str(save_labels_dir)}/{img_dst_path.name.split(".")[0]}.txt'
            with open(label_path, mode = "w") as f:
                aids = coco.gid_to_aids[gid]
                for aid in aids:
                    coco_anno = coco.anns[aid]

                    try:
                        cls = coco_to_yolo_catId[coco_anno["category_id"]]
                    except:
                        continue

                    x, y, w, h = coco_anno["bbox"]
                    dh, dw = (1/img.shape[0], 1/img.shape[1])

                    center_x      = (x + w / 2) * dw
                    center_y      = (y + h / 2) * dh
                    width         = w * dw
                    height        = h * dh

                    center_x = max(0, min(center_x, 1.0))
                    center_y = max(0, min(center_y, 1.0))
                    width    = max(0, min(width, 1.0))
                    height   = max(0, min(height, 1.0))

                    annotations = [cls, center_x, center_y, width, height]

                    if task == "keypoint":
                        if "keypoints" in coco_anno:
                            kpts = coco_anno["keypoints"]
                            kpts = [kpts[idx:idx + 3] for idx in range(0, len(kpts), 3)]
                            kpts = [[kp[0]*dw, kp[1]*dh, kp[2]] for kp in kpts]
                            kpts = [i for kp in kpts for i in kp]
                            annotations += kpts

                    if task == "segmentation":
                        if "segmentation" in coco_anno:
                            segs = coco_anno["segmentation"]
                            segs = [seg[idx:idx + 2] for seg in segs for idx in range(0, len(seg), 2)]
                            segs = [[seg[0]*dw, seg[1]*dh] for seg in segs]
                            segs = [s for seg in segs for s in seg]
                            annotations += segs

                    print(*annotations, file=f)

    return config_path

def imread_web(url):
    res = requests.get(url)
    img = None
    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
    return img
