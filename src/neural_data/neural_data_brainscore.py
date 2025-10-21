import pandas as pd


def load_majajhong2015_data(region, avg_trials=True, subject_name=None, **kwargs):

    import brainscore_vision.benchmarks.majajhong2015.benchmark as majajhong2015
    from brainscore_vision.benchmark_helpers.screen import place_on_screen

    neural_data = majajhong2015.load_assembly(average_repetitions=False, region=region, access='public')
    visual_degree = 8

    img_paths = place_on_screen(neural_data.stimulus_set,
                                target_visual_degrees=8,
                                source_visual_degrees=visual_degree).stimulus_paths

    df = neural_data.to_pandas()
    img_paths = pd.DataFrame().from_dict(img_paths, orient='index', columns=['image_path'])

    # Both img_paths and df must have same number of levels in their columns
    df.columns = df.columns.get_level_values(0)
    df = df.join(img_paths, on='image_id')
    df = df.set_index('image_path', append=True)

    # Check if image paths match
    for fname, path in zip(df.index.get_level_values('filename'),  df.index.get_level_values('image_path')):
        path = path.stem + '.png'
        assert fname == path, f"image paths do not match: {fname} != {path}"

    # Keep only the desired index levels and repetition
    prev_index_levels = df.index.names
    new_index_levels = ['image_id', 'stimulus', 'object_name', 'category_name', 'image_path']
    index_levels_drop = list(set(prev_index_levels) - set(new_index_levels + ['repetition']))
    df = df.droplevel(level=index_levels_drop)

    # Average over repetitions
    if avg_trials:
        df = df.groupby(new_index_levels).agg('mean')

    df = df.sort_values(by=['image_id'], ascending=[True], ignore_index=False)

    # df has columns "Chabo_L_M_5_9", "Tito_L_M_7_1", ...
    # Change it to multi-index columns with monkey and neuron
    new_columns = []
    for col in df.columns:
        monkey, neuron = col.split('_')[0], col
        new_columns.append((monkey, neuron))
    df.columns = pd.MultiIndex.from_tuples(new_columns, names=["monkey", "neuron"])

    if subject_name is not None:
        assert subject_name in ['Chabo', 'Tito'], f"subject_name {subject_name} must be one of ['Chabo', 'Tito']"
        df = df.iloc[:, df.columns.get_level_values(0) == subject_name]

    return df


def load_freemanziemba2013_data(region, avg_trials=True, **kwargs):

    import brainscore_vision.benchmarks.freemanziemba2013.benchmarks.benchmark as freemanziemba2013
    from brainscore_vision.benchmark_helpers.screen import place_on_screen

    neural_data = freemanziemba2013.load_assembly(average_repetitions=False, region=region, access='public')
    visual_degree = 4

    img_paths = place_on_screen(neural_data.stimulus_set,
                                target_visual_degrees=8,
                                source_visual_degrees=visual_degree).stimulus_paths

    df = neural_data.to_pandas()
    img_paths = pd.DataFrame().from_dict(img_paths, orient='index', columns=['image_path'])

    # Both img_paths and df must have same number of levels in their columns
    df.columns = df.columns.get_level_values(0)
    df = df.join(img_paths, on='image_id')
    df = df.set_index('image_path', append=True)

    # Check if image paths match
    for fname, path in zip(df.index.get_level_values('filename'),  df.index.get_level_values('image_path')):
        path = path.split('/')[-1]
        assert fname == path, f"image paths do not match: {fname} != {path}"

    # Keep only the desired index levels and repetition
    prev_index_levels = df.index.names
    new_index_levels = ['image_id', 'texture_family', 'texture_type', 'image_path']
    index_levels_drop = list(set(prev_index_levels) - set(new_index_levels + ['repetition']))
    df = df.droplevel(level=index_levels_drop)

    # Average over repetitions
    if avg_trials:
        df = df.groupby(new_index_levels).agg('mean')

    df = df.sort_values(by=['image_id'], ascending=[True], ignore_index=False)

    return df


def load_img_from_df(df):

    import PIL

    img_dict = {col_name: [] for col_name in df.index.names}
    img_dict['pil_imgs'] = []

    for idx, _ in df.iterrows():

        for i, col_name in enumerate(df.index.names):
            img_dict[col_name].append(idx[i])

        img_path = idx[-1]  # Last element of the index is the image path
        pil_img = PIL.Image.open(img_path)
        img_dict['pil_imgs'].append(pil_img)

    img_df = pd.DataFrame().from_dict(img_dict, orient='columns').set_index(df.index.names[:-1])

    return img_df
