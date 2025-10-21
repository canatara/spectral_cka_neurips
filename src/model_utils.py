import os
import numpy as np

import torch
import timm

from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader


DATA_ROOT = './data/'
ACTIVATIONS_ROOT = './activations/'


class ImageDatasetFromDataFrame(Dataset):
    def __init__(self, df_img, transform):
        self.df_img = df_img
        self.transform = transform

        # Determine which column to use for images
        if 'pil_imgs' in self.df_img.columns:
            self.use_pil = True
        elif 'image_path' in self.df_img.columns:
            self.use_pil = False
        else:
            raise ValueError("DataFrame must contain either 'pil_imgs' or 'image_path' column.")

    def __len__(self):
        return len(self.df_img)

    def __getitem__(self, idx):
        try:
            if self.use_pil:
                img = self.df_img.iloc[idx]['pil_imgs']
            else:
                img_path = self.df_img.iloc[idx]['image_path']
                img = read_image(str(img_path), ImageReadMode.RGB)
                img = img.float() / 255.0

            return self.transform(img)

        except Exception as e:
            raise Exception(f"Error loading image at index {idx}: {e}")


def get_activations(model_name, images, dataset_name,
                    pretrained=True, device='cuda', batch_size=500,
                    activations_root=ACTIVATIONS_ROOT):

    activations_root = os.path.join(activations_root, dataset_name)
    os.makedirs(activations_root, exist_ok=True)

    if pretrained:
        model_id = f"{model_name}_pretrained"
    else:
        model_id = f"{model_name}_random"

    filename = os.path.join(activations_root, f"{model_id}_{dataset_name}.npz")
    if os.path.exists(filename):
        print(f'Loading activations from file {filename}')
        data = np.load(filename)
        activations = {key: data[key] for key in data.keys()}

    else:
        print('Extracting activations...')
        model = timm.create_model(model_name, features_only=True, pretrained=pretrained)
        model = model.eval().type(images.dtype).to(device)
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)

        feat_names = model.feature_info.module_name()
        print(feat_names)

        image_tensor = transform(images)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        activations = {key: [] for key in feat_names}
        for n in range(image_tensor.shape[0]//batch_size+1):

            imgs_batch = image_tensor[n*batch_size:(n+1)*batch_size]
            with torch.no_grad():  # Disable gradient computation for efficiency
                output = model(imgs_batch.to(device))
                for feat, act in zip(feat_names, output):
                    activations[feat] += [act.cpu().detach()]

        for key in activations.keys():
            activations[key] = torch.cat(activations[key], axis=0).numpy()

        print('Finished...')

        np.savez(filename, **activations)

    return activations


def get_activations_from_df_img(df_img, model_name, dataset_name,
                                pretrained=True, device='cuda', batch_size=32,
                                activations_root=ACTIVATIONS_ROOT, num_workers=4):
    """
    Optimized version that uses DataLoader for better efficiency

    Parameters:
    -----------
    df_img : pandas.DataFrame
        DataFrame containing image paths (assumes 'image_path' or 'pil_imgs' column)
    model_name : str
        Name of the model to use
    dataset_name : str
        Name of the dataset for saving activations
    pretrained : bool
        Whether to use pretrained weights
    device : str
        Device to run the model on
    batch_size : int
        Batch size for processing (smaller for memory efficiency)
    activations_root : str
        Root directory to save/load activations
    num_workers : int
        Number of workers for data loading

    Returns:
    --------
    activations : dict
        Dictionary containing activations for each layer
    """

    activations_root = os.path.join(activations_root, dataset_name)
    os.makedirs(activations_root, exist_ok=True)

    if pretrained:
        model_id = f"{model_name}_pretrained"
    else:
        model_id = f"{model_name}_random"

    filename = os.path.join(activations_root, f"{model_id}_{dataset_name}.npz")
    if os.path.exists(filename):
        print(f'Loading activations from file {filename}')
        data = np.load(filename)
        activations = {key: data[key] for key in data.keys()}

    else:
        print('Extracting activations from DataFrame (optimized)...')
        model = timm.create_model(model_name, features_only=True, pretrained=pretrained).eval().to(device)
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)

        feat_names = model.feature_info.module_name()
        print(f"Feature names: {feat_names}")

        # Create dataset and dataloader
        dataset = ImageDatasetFromDataFrame(df_img, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)

        activations = {key: [] for key in feat_names}

        print("Processing batches...")
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():  # Disable gradient computation for efficiency
                output = model(batch.to(device))
                for feat, act in zip(feat_names, output):
                    activations[feat].append(act.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

        for key in activations.keys():
            activations[key] = torch.cat(activations[key], axis=0).numpy()

        print('Finished extracting activations...')
        np.savez(filename, **activations)

    return activations


def get_class_sorted_data(resp, imgs, istim, classes, class_names, debug=False):

    sorted_responses = []
    sorted_imgs = []
    sorted_classes = []
    sorted_istim = []
    sorted_idxs = []

    for i in range(12):

        sort_idx = np.where(classes == i)[0]
        sort_idx = sort_idx[np.argsort(istim[sort_idx])]

        sort_resp = resp[sort_idx]
        sort_imgs = imgs[sort_idx]
        sort_classes = classes[sort_idx]
        sort_istim = istim[sort_idx]

        sorted_responses += [sort_resp]
        sorted_imgs += [sort_imgs]
        sorted_classes += [sort_classes]
        sorted_istim += [sort_istim]
        sorted_idxs += [sort_idx]

        if debug:

            import matplotlib.pyplot as plt

            print(sort_resp.shape, sort_imgs.shape, sort_classes.shape, np.unique(sort_istim).shape)

            resp1 = sort_resp[:50].reshape(50, -1).astype(np.float64)
            # resp1 = sort_imgs[:50].reshape(50, -1).astype(np.float64)
            resp1 -= resp1.mean(axis=1, keepdims=True)
            resp1 /= np.linalg.norm(resp1, axis=1, keepdims=True)

            fig, axs = plt.subplots(1, 6, figsize=(10, 4))
            for n, j in enumerate(np.random.randint(0, len(np.unique(sort_imgs, axis=0)), 5)):
                axs[n].imshow(np.unique(sort_imgs, axis=0)[j])
                axs[n].set_title(f"Class: {class_names[i]}")
                axs[n].axis('off')

            axs[5].imshow(resp1@resp1.T/resp1.shape[1])
            axs[5].axis('off')
            plt.show()

    return sorted_responses, sorted_imgs, sorted_classes, sorted_istim, sorted_idxs
