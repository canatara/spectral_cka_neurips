from .neural_data import neural_data_brainscore as brainscore
from .neural_data import neural_data_stringer as stringer
from .neural_data import neural_data_tvsd as tvsd
from .neural_data import neural_data_things_fmri as things


def get_stringer(DATASET_ROOT, avg_trials=True, debug=False):

    df, df_img = stringer.load_stringer_data(DATASET_ROOT, avg_trials=avg_trials, debug=debug)

    return df, df_img


def get_brainscore(region, avg_trials=True, get_stimuli=False, **kwargs):

    if region in ['IT', 'V4']:
        df = brainscore.load_majajhong2015_data(region, avg_trials, **kwargs)

    elif region in ['V1', 'V2']:
        df = brainscore.load_freemanziemba2013_data(region, avg_trials, **kwargs)

    else:
        raise Exception(f"Only {['majajhong2015', 'freemanziemba2013']} allowed.")

    if get_stimuli:
        df_img = brainscore.load_img_from_df(df)
    else:
        import pandas as pd

        df_img = pd.DataFrame()
        df_img.index = df.index
        df_img = df_img.reset_index(level='image_path')  # Return only image paths

    return df, df_img


def get_tvsd(DATASET_ROOT, subject_name, region, session='train', get_stimuli=False):

    df = tvsd.load_tvsd_dataset(DATASET_ROOT, subject_name, region, session)

    if get_stimuli:
        df_img = tvsd.load_img_from_df(DATASET_ROOT, df)  # Only returns the image paths
    else:
        df_img = None

    return df, df_img


def get_things(DATASET_ROOT, subject_name, region, get_stimuli=False):

    df = things.load_fMRI_data(DATASET_ROOT, subject_name, region)

    if get_stimuli:
        df_img = things.load_img_from_df(DATASET_ROOT, df)  # Only returns the image paths
    else:
        df_img = None

    return df, df_img


def tensor_imgs_from_df(df_img):

    import torch
    import torchvision.transforms.functional as transform

    tensor_labels = []
    tensor_imgs = []
    for idx, d in df_img.iterrows():

        pil_img = d['pil_imgs']
        tensor_img = transform.to_tensor(pil_img).unsqueeze(0).type(torch.float32)

        tensor_imgs += [tensor_img]
        tensor_labels += [idx]

    tensor_imgs = torch.cat(tensor_imgs)

    return tensor_labels, tensor_imgs


def get_cifar10(DATASET_ROOT):

    import torch
    import torchvision.datasets as datasets

    trainset = datasets.CIFAR10(DATASET_ROOT, train=False,
                                download=True, transform=None)

    imgs = torch.tensor(trainset.data, dtype=torch.float32).permute(0, 3, 1, 2)

    return imgs
