import numpy as np

from pathlib import Path
from scipy import ndimage

from cellori.utils.transforms import batch_normalize, batch_standardize, RandomAugment, subpixel_distance_transform


def load_datasets(path, adjustment='normalize'):

    train = {'images': [], 'coords': []}
    valid = {'images': [], 'coords': []}
    test = {'images': [], 'coords': []}

    path = Path(path)
    if path.is_file() and path.suffix == '.npz':
        datasets = [path]
    else:
        datasets = path.glob('*.npz')

    for npz in datasets:

        data = np.load(npz, allow_pickle=True)

        train['images'].append(data['x_train'])
        train['coords'].append(data['y_train'])
        valid['images'].append(data['x_valid'])
        valid['coords'].append(data['y_valid'])
        test['images'].append(data['x_test'])
        test['coords'].append(data['y_test'])

    train['images'] = np.concatenate(train['images'])
    train['coords'] = np.concatenate(train['coords'])
    valid['images'] = np.concatenate(valid['images'])
    valid['coords'] = np.concatenate(valid['coords'])
    test['images'] = np.concatenate(test['images'])
    test['coords'] = np.concatenate(test['coords'])

    if adjustment == 'normalize':
        train['images'] = np.asarray(batch_normalize(train['images']))
        valid['images'] = np.asarray(batch_normalize(valid['images']))
        test['images'] = np.asarray(batch_normalize(test['images']))
    elif adjustment == 'standardize':
        train['images'] = np.asarray(batch_standardize(train['images']))
        valid['images'] = np.asarray(batch_standardize(valid['images']))
        test['images'] = np.asarray(batch_standardize(test['images']))

    ds = {
        'train': train,
        'valid': valid,
        'test': test
    }

    return ds


def transform_dataset(ds, key, output_shape=(256, 256), min_spots=3):

    base_scales = np.ones(len(ds['images']))

    # Create transformer
    transformer = RandomAugment()
    transformer.generate_transforms(ds['images'], key, base_scales, output_shape)

    # Apply transformations
    images = transformer.apply_image_transforms(ds['images'], interpolation='bilinear')
    coords_list = transformer.apply_coord_transforms(ds['coords'], filter_coords=True)

    # Remove images with less than min_spots
    counts = np.array([len(coord) for coord in coords_list])
    pop_indices = np.where((counts < min_spots))[0]
    for index in np.flip(pop_indices):
        images.pop(index)
        coords_list.pop(index)

    coords = np.empty(len(coords_list), dtype=object)
    coords[:] = coords_list

    transformed_ds = {
        'images': np.array(images)[:, :, :, None],
        'coords': coords,
    }

    return transformed_ds


def transform_batch(batch, coords_pad_length=None, num_label_dilations=1):

    output_shape = batch['images'].shape[1:3]
    coords = batch.pop('coords')

    deltas, labels, _ = subpixel_distance_transform(coords, coords_pad_length, output_shape)
    dilated_labels = np.array(labels)

    if num_label_dilations > 0:
        for i in range(len(dilated_labels)):
            dilated_labels[i] = ndimage.binary_dilation(dilated_labels[i], structure=np.ones((3, 3, 1), dtype=bool),
                                                        iterations=num_label_dilations)

    batch['deltas'] = deltas
    batch['labels'] = labels
    batch['dilated_labels'] = dilated_labels

    return batch
