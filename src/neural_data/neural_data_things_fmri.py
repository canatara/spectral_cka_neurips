import os
from typing import List, Tuple

import pandas as pd
from torch.utils.data import Dataset


def load_fMRI_data(DATASET_ROOT, subject_name, roi):

    FMRI_DATA_ROOT = os.path.join(DATASET_ROOT, 'THINGS', 'fMRI')
    filename = f'processed_sub-{subject_name}_{roi}.pkl'
    filepath = os.path.join(FMRI_DATA_ROOT, filename)

    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
    else:
        df = _load_fMRI_data(DATASET_ROOT, subject_name, roi)
        df.to_pickle(filepath)

    return df


def _load_fMRI_data(DATASET_ROOT, subject_name, roi):

    FMRI_DATA_ROOT = os.path.join(DATASET_ROOT, 'THINGS', 'fMRI')

    stim_file = os.path.join(FMRI_DATA_ROOT, f'sub-{subject_name}_StimulusMetadata.csv')
    voxel_file = os.path.join(FMRI_DATA_ROOT, f'sub-{subject_name}_VoxelMetadata.csv')
    data_file = os.path.join(FMRI_DATA_ROOT, f'sub-{subject_name}_ResponseData.h5')

    voxdata = pd.read_csv(voxel_file)
    stim_data = pd.read_csv(stim_file)
    responses = pd.read_hdf(data_file)
    print(responses.shape, stim_data.shape, voxdata.shape)

    # Extract the voxel ids for specified ROI
    # Check ROI info:
    # https://www.oxcns.org/papers/655%20Rolls%20et%20al%202022%20Human%20posterior%20parietal%20cortex%20Supp%20Mat.pdf
    voxels = voxdata[voxdata[roi] == 1]
    voxel_ids = voxels['voxel_id'].to_numpy()

    # Extract data corresponding to desired voxel ids
    voxel_responses = responses[responses.index.isin(voxel_ids)]
    voxel_responses = voxel_responses.drop(columns=['voxel_id']).T
    voxel_dimension = voxel_responses.shape[1]

    # Read stim data and generate dataframe
    images_with_fMRI = set(stim_data['stimulus'])
    concepts_with_fMRI = set(stim_data['concept'])

    print(voxel_responses.shape, stim_data.shape)
    final_df = stim_data.join(voxel_responses)
    # final_df = final_df.drop(columns=['session', 'run', 'subject_id', 'trial_id'])
    final_df = final_df.drop(columns=['trial_type'])
    final_df = final_df.set_index(["concept", "stimulus", "session", "run", "trial_id", 'subject_id']).sort_index()
    assert final_df.shape[1] == voxel_dimension, f'{final_df.shape[1]} != {voxel_dimension}'

    concepts = final_df.index.levels[0].to_list()
    assert set(final_df.index.levels[1]) == images_with_fMRI
    assert set(concepts) == concepts_with_fMRI

    return final_df


def load_img_from_df(DATASET_ROOT, df):

    THINGS_IMG_PATH = os.path.join(DATASET_ROOT, 'THINGS', 'images', 'imgs_train')
    assert os.path.exists(THINGS_IMG_PATH), f'THINGS images not found in {THINGS_IMG_PATH}'

    img_dict = {col_name: [] for col_name in df.index.names}
    img_dict['pil_imgs'] = []

    for idx, _ in df.iterrows():

        for i, col_name in enumerate(df.index.names):
            img_dict[col_name].append(idx[i])

        category_name, object_name = idx[0], idx[1]
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


def download_things_data(DATASET_ROOT):

    # Download THINGS-data: fMRI Single Trial Responses (table format)
    # https://plus.figshare.com/articles/dataset/THINGS-data_fMRI_Single_Trial_Responses_table_format_/20492835
    url = 'https://plus.figshare.com/ndownloader/files/43635873'

    import zipfile
    import urllib

    # Path to save the downloaded zip file
    zip_path = os.path.join(DATASET_ROOT, "./betas_csv.zip")

    # Download the zip file
    urllib.request.urlretrieve(url, zip_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_ROOT)

    # Rename the extracted folder
    old_folder_name = os.path.join(DATASET_ROOT, "betas_csv")
    new_folder_name = os.path.join(DATASET_ROOT, "THINGS_fMRI")
    os.rename(old_folder_name, new_folder_name)


ALL_ROIS = ['V1', 'V2', 'V3',
            'hV4', 'VO1', 'VO2',
            'LO1 (prf)', 'LO2 (prf)',
            'TO1', 'TO2', 'V3b', 'V3a',
            'lEBA', 'rEBA', 'lFFA', 'rFFA',
            'lOFA', 'rOFA', 'lSTS', 'rSTS',
            'lPPA', 'rPPA', 'lRSC', 'rRSC',
            'lTOS', 'rTOS', 'lLOC', 'rLOC',
            'IT', 'glasser-V1', 'glasser-MST',
            'glasser-V6', 'glasser-V2', 'glasser-V3',
            'glasser-V4', 'glasser-V8', 'glasser-4',
            'glasser-3b', 'glasser-FEF', 'glasser-PEF',
            'glasser-55b', 'glasser-V3A', 'glasser-RSC',
            'glasser-POS2', 'glasser-V7', 'glasser-IPS1',
            'glasser-FFC', 'glasser-V3B', 'glasser-LO1',
            'glasser-LO2', 'glasser-PIT', 'glasser-MT',
            'glasser-A1', 'glasser-PSL', 'glasser-SFL',
            'glasser-PCV', 'glasser-STV', 'glasser-7Pm',
            'glasser-7m', 'glasser-POS1', 'glasser-23d',
            'glasser-v23ab', 'glasser-d23ab', 'glasser-31pv',
            'glasser-5m', 'glasser-5mv', 'glasser-23c',
            'glasser-5L', 'glasser-24dd', 'glasser-24dv',
            'glasser-7AL', 'glasser-SCEF', 'glasser-6ma',
            'glasser-7Am', 'glasser-7Pl', 'glasser-7PC',
            'glasser-LIPv', 'glasser-VIP', 'glasser-MIP',
            'glasser-1', 'glasser-2', 'glasser-3a',
            'glasser-6d', 'glasser-6mp', 'glasser-6v',
            'glasser-p24pr', 'glasser-33pr', 'glasser-a24pr',
            'glasser-p32pr', 'glasser-a24', 'glasser-d32',
            'glasser-8BM', 'glasser-p32', 'glasser-10r',
            'glasser-47m', 'glasser-8Av', 'glasser-8Ad',
            'glasser-9m', 'glasser-9p', 'glasser-10d',
            'glasser-8C', 'glasser-44', 'glasser-45',
            'glasser-47l', 'glasser-a47r', 'glasser-6r',
            'glasser-IFJa', 'glasser-IFJp', 'glasser-IFSp',
            'glasser-IFSa', 'glasser-p9-46v', 'glasser-46',
            'glasser-a9-46v', 'glasser-9-46d', 'glasser-9a',
            'glasser-10v', 'glasser-a10p', 'glasser-10pp',
            'glasser-11l', 'glasser-13l', 'glasser-OFC',
            'glasser-47s', 'glasser-LIPd', 'glasser-6a',
            'glasser-i6-8', 'glasser-s6-8', 'glasser-43',
            'glasser-OP4', 'glasser-OP1', 'glasser-OP2-3',
            'glasser-52', 'glasser-RI', 'glasser-PFcm',
            'glasser-PoI2', 'glasser-TA2', 'glasser-FOP4',
            'glasser-MI', 'glasser-Pir', 'glasser-AVI',
            'glasser-AAIC', 'glasser-FOP1', 'glasser-FOP3',
            'glasser-FOP2', 'glasser-PFt', 'glasser-AIP',
            'glasser-EC', 'glasser-PreS', 'glasser-H',
            'glasser-ProS', 'glasser-PeEc', 'glasser-STGa',
            'glasser-PBelt', 'glasser-A5', 'glasser-PHA1',
            'glasser-PHA3', 'glasser-STSda', 'glasser-STSdp',
            'glasser-STSvp', 'glasser-TGd', 'glasser-TE1a',
            'glasser-TE1p', 'glasser-TE2a', 'glasser-TF',
            'glasser-TE2p', 'glasser-PHT', 'glasser-PH',
            'glasser-TPOJ1', 'glasser-TPOJ2', 'glasser-TPOJ3',
            'glasser-DVT', 'glasser-PGp', 'glasser-IP2',
            'glasser-IP1', 'glasser-IP0', 'glasser-PFop',
            'glasser-PF', 'glasser-PFm', 'glasser-PGi',
            'glasser-PGs', 'glasser-V6A', 'glasser-VMV1',
            'glasser-VMV2', 'glasser-PHA2', 'glasser-V4t',
            'glasser-FST', 'glasser-V3CD', 'glasser-LO3',
            'glasser-31pd', 'glasser-31a', 'glasser-VVC',
            'glasser-25', 'glasser-s32', 'glasser-pOFC',
            'glasser-PoI1', 'glasser-Ig', 'glasser-FOP5',
            'glasser-p10p', 'glasser-p47r', 'glasser-TGv',
            'glasser-MBelt', 'glasser-LBelt', 'glasser-A4',
            'glasser-STSva', 'glasser-TE1m', 'glasser-PI',
            'glasser-a32pr', 'glasser-p24']
