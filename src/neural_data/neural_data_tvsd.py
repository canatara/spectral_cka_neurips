import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset

# See https://gin.g-node.org/paolo_papale/TVSD for more information
# The processed data is available at:
# monkey N: https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyN/THINGS_normMUA.mat
# monkey F: https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyF/THINGS_normMUA.mat


def load_tvsd_dataset(DATASET_ROOT, monkey_name, roi, session='train'):

    assert monkey_name in ['F', 'N'], 'Invalid monkey name'
    assert roi in ['V1', 'V4', 'IT'], 'Invalid roi'

    DATA_ROOT_TVSD = os.path.join(DATASET_ROOT, 'TVSD')

    all_data_path = os.path.join(DATA_ROOT_TVSD, f'tvsd_processed_{monkey_name}_{session}.npz')
    if os.path.exists(all_data_path):
        # Load data_dict
        data_dict = np.load(all_data_path, allow_pickle=True)['data_dict'].tolist()
    else:
        # Generate data_dict
        # os.makedirs(DATA_ROOT_TVSD, exist_ok=True)
        data_dict = _load_tvsd_dataset(DATASET_ROOT, monkey_name, session)
        np.savez(all_data_path, data_dict=data_dict)

    df = data_dict['df'].copy()
    df = df.iloc[:, df.columns.get_level_values(0) == roi]

    return df


def _load_tvsd_dataset(DATASET_ROOT, monkey_name, session='train'):

    import mat73

    DATA_ROOT_TVSD = os.path.join(DATASET_ROOT, 'TVSD')
    if not os.path.exists(DATA_ROOT_TVSD):
        print('Downloading data...')
        download_tvsd_data(DATASET_ROOT)

    datapath = os.path.join(DATA_ROOT_TVSD, f'monkey{monkey_name}')

    mat_data = mat73.loadmat(os.path.join(datapath, 'THINGS_normMUA.mat'))
    image_labels = mat73.loadmat(os.path.join(datapath, '_logs', 'things_imgs.mat'))

    if session == 'test':
        MUA = np.asarray(mat_data[f'{session}_MUA_reps'])
    else:
        MUA = np.asarray(mat_data[f'{session}_MUA'])
    SNR = np.array(mat_data['SNR'])
    arrayID = np.repeat(np.arange(16), 64)

    things_path = np.asarray(image_labels[f'{session}_imgs']['things_path'])
    classes = np.asarray(image_labels[f'{session}_imgs']['class'])

    category_name = np.asarray([obj.split('\\')[0] for obj in things_path])
    object_name = np.asarray([obj.split('\\')[-1] for obj in things_path])
    assert np.all(category_name == classes), 'All category names should be the same'

    # For monkey N, this plot should show that array 5 is empty.
    # So we need to create a mask for that monkey
    if monkey_name == 'N':

        rois = np.zeros(1024)  # V1
        rois[512:768] = 1  # V4
        rois[768:] = 2  # IT

        validArrays = arrayID != 5
        rois = rois[validArrays]
        arrayID = np.repeat(np.arange(15), 64)
        SNR = SNR[validArrays, :]
        MUA = MUA[validArrays, :]

    if monkey_name == 'F':

        rois = np.zeros(1024)  # ; % V1
        rois[512:832] = 2  # ; % IT
        rois[832:] = 1  # ; % V4

        idx = np.arange(1024)
        idx = np.concatenate([idx[rois == 0], idx[rois == 1], idx[rois == 2]])
        MUA = MUA[idx, :]
        rois = rois[idx]
        # DO NOT REORDER ARRAY ID.

    V1 = MUA[rois == 0, :].swapaxes(0, 1)  # shape (num_stim, num_neurons)
    V4 = MUA[rois == 1, :].swapaxes(0, 1)  # shape (num_stim, num_neurons)
    IT = MUA[rois == 2, :].swapaxes(0, 1)  # shape (num_stim, num_neurons)

    data_dict = {'category_name': category_name,
                 'object_name': object_name}

    data_dict |= {f'V1_{n+1}': V1[:, n] for n in range(V1.shape[1])}
    data_dict |= {f'V4_{n+1}': V4[:, n] for n in range(V4.shape[1])}
    data_dict |= {f'IT_{n+1}': IT[:, n] for n in range(IT.shape[1])}

    df = pd.DataFrame().from_dict(data_dict, orient='columns')
    df = df.set_index(['category_name', 'object_name'], inplace=False).sort_index()

    # df has columns "V1_1", "V1_2", ..., "V4_1", "V4_2", ..., "IT_1", "IT_2", ...
    # Change it to multi-index columns with roi and neuron
    new_columns = []
    for col in df.columns:
        roi, neuron = col.split('_')
        new_columns.append((roi, neuron))
    df.columns = pd.MultiIndex.from_tuples(new_columns, names=["roi", "neuron"])

    returns = dict(df=df, arrayID=arrayID, SNR=SNR, rois=rois)

    return returns


def load_img_from_df(DATASET_ROOT, df):

    THINGS_IMG_PATH = os.path.join(DATASET_ROOT, 'THINGS', 'images', 'imgs_train')
    assert os.path.exists(THINGS_IMG_PATH), f'THINGS images not found in {THINGS_IMG_PATH}'

    img_dict = {col_name: [] for col_name in df.index.names}
    img_dict['pil_imgs'] = []

    for idx, _ in df.iterrows():

        for i, col_name in enumerate(df.index.names):
            img_dict[col_name].append(idx[i])

        category_name, object_name = idx
        img_path = os.path.join(THINGS_IMG_PATH, category_name, object_name)
        # pil_img = PIL.Image.open(img_path)
        img_dict['pil_imgs'].append(img_path)

    img_df = pd.DataFrame().from_dict(img_dict, orient='columns')

    return img_df


class THINGSDataset(Dataset):

    def __init__(self, root, path_dict, transform=None, target_transform=None, only_img=True):
        """
        Input must be of form path_dict = {concept: [path1, path2, ...], ...}

        Followed this link: https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html
        """

        self.classes = sorted(list(path_dict.keys()))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self.make_dataset(root, path_dict)
        self.targets = [s[1] for s in self.samples]

        self.transform = transform
        self.target_transform = target_transform
        self.only_img = only_img

    def make_dataset(self, root, path_dict) -> List[Tuple[str, int]]:

        instances = []
        for target_class in sorted(self.class_to_idx.keys()):

            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)

            for fname in sorted(path_dict[target_class]):
                path = os.path.join(target_dir, fname)
                item = path, class_index
                instances.append(item)

        return instances

    def __getitem__(self, index):

        from torchvision.datasets.folder import default_loader

        path, target = self.samples[index]

        # sample = get_sample(path)
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.only_img:
            return sample
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


def get_sample(path):

    from PIL import Image
    with open(path, "rb") as f:
        img = Image.open(f)
        sample = img.convert("RGB")

    return sample


def download_tvsd_data(DATASET_ROOT):

    # Dataset link:
    # https://gin.g-node.org/paolo_papale/TVSD/archive/master.zip
    url = "https://gin.g-node.org/paolo_papale/TVSD/archive/master.zip"

    import zipfile
    import urllib

    # Path to save the downloaded zip file
    zip_path = os.path.join(DATASET_ROOT, "./master.zip")

    # Download the zip file
    urllib.request.urlretrieve(url, zip_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_ROOT)

    # Rename the extracted folder
    old_folder_name = os.path.join(DATASET_ROOT, "tvsd")
    new_folder_name = os.path.join(DATASET_ROOT, "TVSD")
    os.rename(old_folder_name, new_folder_name)

    # Next download the processed data for both monkeys
    monkeyN = 'https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyN/THINGS_normMUA.mat'
    monkeyF = 'https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyF/THINGS_normMUA.mat'

    urllib.request.urlretrieve(monkeyN, os.path.join(new_folder_name, "monkeyN/THINGS_normMUA.mat"))
    urllib.request.urlretrieve(monkeyF, os.path.join(new_folder_name, "monkeyF/THINGS_normMUA.mat"))
