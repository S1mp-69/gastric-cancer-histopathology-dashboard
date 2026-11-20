import numpy as np
import spams


def rgb2od(img, Io=255):
    img = img.astype(np.float)
    img = np.clip(img, 1e-4, 255)
    od = -np.log(img / Io)
    return od


def od2rgb(od, Io=255):
    od = np.clip(od, 1e-4, None)
    rgb = np.multiply(Io, np.exp(-od))
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)


def get_intensities(W, od):
    H = np.linalg.lstsq(W, od.T, rcond=None)[0]
    H = np.clip(H, 0, None)

    return H


def mh_update(prev, gamma, mu, sigma):
    # gaussian metropolis-hastings update
    # TODO: implement other distributions
    cand = prev + np.random.normal(0.0, gamma)
    p = np.min([1.0, np.exp(np.sum((prev - mu)**2 - (cand - mu)**2) / sigma**2)])
    u = np.random.uniform(0, 1)
    return cand if u < p else prev


def get_mask(od, beta1, beta2):
    mask = ~((np.sum(od ** 2, axis=1) > beta1 ** 2) & (np.sum(od ** 2, axis=1) < beta2 ** 2))

    return mask


def reconstruct_image(W, H, od, beta1, beta2, size):
    if W is None:
        return (H * np.ones(size)).astype(np.uint8)
    new_od = (W @ H).T
    if od is not None:
        mask = get_mask(od, beta1, beta2)
        new_od[mask] = od[mask]
    new_od = new_od.reshape(size)
    img = od2rgb(new_od)
    return img


# def split_data_by_hue(images, labels, hue_sorted_idx, n_chunk):
#     hue_sorted_images = images[hue_sorted_idx]
#     hue_sorted_labels = labels[hue_sorted_idx]
#     pos_images = hue_sorted_images[hue_sorted_labels == 1]
#     neg_images = hue_sorted_images[hue_sorted_labels == 0]
#     n_pos = len(pos_images)
#     n_neg = len(neg_images)
#     k_pos = n_pos // n_chunk
#     k_neg = n_neg // n_chunk
#     hue_split_data = []
#     for i in range(n_chunk):
#         pos_chunk_images = pos_images[i * k_pos:(i + 1) * k_pos]
#         pos_chunk_labels = np.ones(len(pos_chunk_images))
#         neg_chunk_images = neg_images[i * k_neg:(i + 1) * k_neg]
#         neg_chunk_labels = np.zeros(len(neg_chunk_images))
#         chunk_images = np.vstack([pos_chunk_images, neg_chunk_images])
#         chunk_labels = np.hstack([pos_chunk_labels, neg_chunk_labels])
#         hue_split_data.append((chunk_images, chunk_labels))
#     return hue_split_data


def get_images_labels(data):
    images, labels = [], []
    for i, l in data:
        images.append(i)
        labels.append(l)
    images = np.vstack(images)
    labels = np.hstack(labels)
    return images, labels


def split_data_by_hue(images, labels, hue_sorted_idx, n_chunk):
    k = len(images) // n_chunk
    hue_sorted_images = images[hue_sorted_idx]
    hue_sorted_labels = labels[hue_sorted_idx]
    hue_split_data = []
    for i in range(n_chunk):
        chunk_images = hue_sorted_images[i * k:(i + 1) * k]
        chunk_labels = hue_sorted_labels[i * k:(i + 1) * k]
        hue_split_data.append((chunk_images, chunk_labels))
    return hue_split_data

def split_array_by_hue(array, hue_sorted_idx, n_chunk):
    array = np.array(array)
    k = len(array) // n_chunk
    hue_sorted_array = array[hue_sorted_idx]
    hue_split_arrays = []
    for i in range(n_chunk):
        chunk_array = hue_sorted_array[i * k:(i + 1) * k]
        hue_split_arrays.append(chunk_array)
    return hue_split_arrays
