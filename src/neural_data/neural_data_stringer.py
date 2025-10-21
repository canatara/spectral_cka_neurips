import numpy as np
import pandas as pd
import os


def load_stringer_data(DATASET_ROOT, avg_trials=True, debug=False):

    import scipy as sp
    from scipy.sparse.linalg import eigsh

    import torch
    import torchvision.transforms.functional as transform

    DATA_ROOT_STRINGER = os.path.join(DATASET_ROOT, 'stringer_data/')
    if not os.path.exists(DATA_ROOT_STRINGER):
        print('Downloading data...')
        download_stringer_data(DATA_ROOT_STRINGER)

    filename = 'natimg2800_M170714_MP032_2017-09-14'  # Each stimuli has 2 or 3 repetitions
    filename = 'natimg2800_M170714_MP032_2017-08-07'  # Each stimuli has 2 repetition

    dat_file_path = os.path.join(DATA_ROOT_STRINGER, f"{filename}.mat")
    class_file_path = os.path.join(DATA_ROOT_STRINGER, 'stimuli_class_assignment_confident.mat')
    img_file_path = os.path.join(DATA_ROOT_STRINGER, 'images_natimg2800_all.mat')

    print('Processing data...')

    # Process neural data
    dat_file = sp.io.loadmat(dat_file_path)
    resp = dat_file['stim'][0]['resp'][0]  # stim x neurons
    spont = dat_file['stim'][0]['spont'][0]  # timepts x neurons
    istim = (dat_file['stim'][0]['istim'][0]).astype(np.int32)  # stim ids
    istim -= 1  # get out of MATLAB convention
    istim = istim[:, 0]
    nimg = istim.max()  # these are blank stims (exclude them)
    resp = resp[istim < nimg, :]
    istim = istim[istim < nimg]

    # subtract spont (32D)
    mu = spont.mean(axis=0)
    sd = spont.std(axis=0) + 1e-6
    resp = (resp - mu) / sd
    spont = (spont - mu) / sd
    sv, u = eigsh(spont.T @ spont, k=32)
    resp = resp - (resp @ u) @ u.T

    # mean center each neuron
    resp -= resp.mean(axis=0)

    # Process class names and assignments
    class_file = sp.io.loadmat(class_file_path)
    classes = class_file['class_assignment'][0]
    classes = classes[istim]
    class_names = class_file['class_names'][0]
    class_names = [c[0] for c in class_names]
    assert len(class_names) == len(set(classes)), "There are 12 classes in total"

    img_file = sp.io.loadmat(img_file_path)
    imgs = np.moveaxis(img_file['imgs'], -1, 0)  # Has shape (68, 270, 2800) for three rotations
    imgs = imgs[:, :, 90:180]  # Only take the second column (shape:(2800, 68, 90))
    imgs = imgs[istim]
    tensor_imgs = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)
    tensor_imgs = tensor_imgs.expand(-1, 3, -1, -1)
    pil_imgs = [transform.to_pil_image(img) for img in tensor_imgs]
    print("Images shape: ", tensor_imgs.shape, classes.shape)

    df_dict = {
        'istim': istim,
        'class': classes,
        'class_name': [class_names[c] for c in classes],
        'pil_imgs': pil_imgs,
    }
    df_dict |= {f'Neuron_{i}': resp[:, i] for i in range(resp.shape[1])}

    df = pd.DataFrame(df_dict)
    df = df.set_index(['class_name', 'class', 'istim'], inplace=False).sort_index()

    df_img = pd.DataFrame()
    df_img['pil_imgs'] = df.pop('pil_imgs')

    if avg_trials:
        df = df.groupby(['class_name', 'class', 'istim']).agg('mean')
        # df_std = df.groupby(['istim', 'class_name']).agg('std')
        df_img = df_img.groupby(['class_name', 'class', 'istim']).agg('sample')

    if debug:
        verify_stringer_data(resp, imgs, istim, classes, class_names)

    return df, df_img


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


def download_stringer_data(DATA_ROOT_STRINGER):

    # Dataset link:
    # https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_in_response_to_2_800_natural_images/6845348

    import zipfile
    import urllib
    import os

    os.makedirs(DATA_ROOT_STRINGER)

    # URL of the zip file
    url = "https://figshare.com/ndownloader/articles/6845348/versions/4"

    # Path to save the downloaded zip file
    zip_path = os.path.join(DATA_ROOT_STRINGER, "./6845348.zip")

    # Download the zip file
    urllib.request.urlretrieve(url, zip_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_ROOT_STRINGER)

    # Download the stimuli class assignment file
    classes_url = (
        'https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/'
        'blob/master/classes/stimuli_class_assignment_confident.mat'
    )
    classes_path = os.path.join(DATA_ROOT_STRINGER, 'stimuli_class_assignment_confident.mat')
    urllib.request.urlretrieve(classes_url, classes_path)


def verify_stringer_data(resp, imgs, istim, classes, class_names):

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 5, figsize=(10, 4))
    for n, i in enumerate(np.random.randint(0, 2800, 5)):
        axs[n].imshow(imgs[i])
        axs[n].set_title(f"Class: {class_names[classes[i]]}")
        axs[n].axis('off')
    plt.show()

    # # sanity check - decent signal variance ?
    # # split stimuli into two repeats
    # NN = resp.shape[1]
    # sresp = np.zeros((2, nimg, NN), np.float64)
    # inan = np.zeros((nimg,), bool)
    # for n in range(nimg):
    #     ist = (istim == n).nonzero()[0]
    #     i1 = ist[:int(ist.size/2)]
    #     i2 = ist[int(ist.size/2):]
    #     # check if two repeats of stim
    #     if np.logical_or(i2.size < 1, i1.size < 1):
    #         inan[n] = 1
    #     else:
    #         sresp[0, n, :] = resp[i1, :].mean(axis=0)
    #         sresp[1, n, :] = resp[i2, :].mean(axis=0)
    # # remove image responses without two repeats
    # sresp = sresp[:, ~inan, :]
    # snorm = sresp - sresp.mean(axis=1)[:, np.newaxis, :]
    # snorm = snorm / (snorm.std(axis=1)[:, np.newaxis, :] + 1e-6)
    # cc = (snorm[0].T @ snorm[1]) / sresp.shape[1]
    # print('fraction of signal variance: %2.3f' % np.diag(cc).mean())

    # # sanity check - decent decoding ?
    # # 1 nearest neighbor decoder
    # # (mean already subtracted)
    # cc = sresp[0] @ sresp[1].T
    # cc /= (sresp[0]**2).sum()
    # cc /= (sresp[1]**2).sum()
    # nstims = sresp.shape[1]
    # print('decoding accuracy: %2.3f' % (cc.argmax(axis=1) == np.arange(0, nstims, 1, int)).mean())

    # ### sanity check - is the powerlaw close to 1 ?
    # # powerlaw
    # # compute cvPCA
    # ss = shuff_cvPCA(sresp)
    # # compute powerlaw of averaged shuffles
    # ss = ss.mean(axis=0)
    # alpha,ypred = get_powerlaw(ss/ss.sum(), np.arange(11,5e2).astype(int))
    # print('powerlaw, alpha=%2.3f'%alpha)
    # plt.loglog(np.arange(0,ss.size)+1, ss/ss.sum())
    # plt.loglog(np.arange(0,ss.size)+1, ypred, c='k')
    # plt.show()
