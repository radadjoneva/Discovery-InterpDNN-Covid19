# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import random
import numpy as np


# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


def split_data(list_idx, seed=42, ratio=[0.7, 0.2, 0.1], split="train"):
    """Split data into training, validation, and test sets based on specified ratios.

    Args:
        data (list): List of file paths or indices.
        seed (int): Seed for the random number generator (default: 42).
        ratio (list): List of three numbers representing the ratio of training, validation, and test sets (default: [0.7, 0.2, 0.1]).
        split (str): The desired split to return ('train', 'val', or 'test').

    Returns:
        list or pd.DataFrame: The specified split (train, val, or test) from the input data.
    """
    random.seed(seed)
    random.shuffle(list_idx)

    n_train = int(ratio[0] * len(list_idx))
    n_val = int(ratio[1] * len(list_idx))

    train_list = list_idx[:n_train]
    val_list = list_idx[n_train : n_train + n_val]
    test_list = list_idx[n_train + n_val :]

    if split == "train":
        return train_list
    elif split == "val":
        return val_list
    else:  # "test"
        return test_list


def select_middle_images(img_paths, patient_id, top_k=10):
    """Select the middle `k` images from a list of image paths.

    Args:
        img_paths (list): List of image paths.
        patient_id (int): Patient ID.
        top_k (int): Number of images to select from the middle.

    Returns:
        list: List of selected image paths.
    """
    num_imgs = len(img_paths)
    if num_imgs >= top_k:
        mid_idx = num_imgs // 2
        start_idx = mid_idx - top_k // 2
        end_idx = start_idx + top_k
        return img_paths[start_idx:end_idx]
    else:
        raise ValueError(f"Patient {patient_id} has less than {top_k} images.")


def select_even_slices(img_paths, nb_imgs=10):
    """Select a subset of CT scan image paths.

    Consider 60% of images, starting from the 20th percentile to the 80th percentile.
    Select 10 evenly spaced images from this subset.

    Args:
        - img_paths (list): List of image paths.
        - nb_imgs (int): Number of images to select.

    Returns:
        - list: List of selected image paths.
    """
    total_images = len(img_paths)

    # Calculate the indices to exclude 20% from the start and 20% from the end
    start_index = int(total_images * 0.2)
    end_index = int(total_images * 0.8)

    # Select the middle 60% of images
    middle_images = img_paths[start_index:end_index]

    # Determine the number of images to select
    nb_imgs = min(nb_imgs, len(middle_images))

    # Select top_k evenly spaced images from the middle 60%
    selected_indices = np.linspace(0, len(middle_images) - 1, nb_imgs, dtype=int)
    selected_images = [middle_images[i] for i in selected_indices]

    return selected_images


# def select_normal_images(img_paths, patient_id, top_k=10):
#     """Select top_k images such that they are approx. normally distributed around the middle image.

#     Args:
#         img_paths (list): List of image paths.
#         patient_id (int): Patient ID.
#         top_k (int): Number of images to select. Defaults to 10.

#     Returns:
#         list: List of selected image paths.
#     """
#     num_imgs = len(img_paths)
#     if num_imgs >= top_k:
#         mean = num_imgs // 2
#         std_dev = top_k // 2  # adjust??

#         # Generate normally distributed indices
#         indices = np.random.normal(loc=mean, scale=std_dev, size=top_k)

#         # Round and clip the indices to be within the valid range
#         indices = np.round(indices).astype(int)
#         indices = np.clip(indices, 0, num_imgs - 1)

#         print(indices)

#         # Select elements based on these indices
#         selected_indices = sorted(list(set(indices)))[:top_k]  # Ensure unique and sorted indices
#         return selected_indices
#         # return [img_paths[i] for i in selected_indices]
#     else:
#         raise ValueError(f"The list has less than {top_k} images.")


if __name__ == "__main__":
    img_paths = [
        "img1.png",
        "img2.png",
        "img3.png",
        "img4.png",
        "img5.png",
        "img6.png",
        "img7.png",
        "img8.png",
        "img9.png",
        "img10.png",
        "img11.png",
        "img12.png",
        "img13.png",
        "img14.png",
        "img15.png",
        "img16.png",
        "img17.png",
        "img18.png",
        "img19.png",
        "img20.png",
        "img21.png",
        "img22.png",
        "img23.png",
        "img24.png",
        "img25.png",
        "img26.png",
        "img27.png",
        "img28.png",
        "img29.png",
        "img30.png",
        "img31.png",
        "img32.png",
        "img33.png",
        "img34.png",
        "img35.png",
        "img36.png",
        "img37.png",
        "img38.png",
        "img39.png",
        "img40.png",
        "img41.png",
        "img42.png",
        "img43.png",
        "img44.png",
        "img45.png",
        "img46.png",
        "img47.png",
        "img48.png",
        "img49.png",
        "img50.png",
        "img51.png",
        "img52.png",
        "img53.png",
        "img54.png",
        "img55.png",
        "img56.png",
        "img57.png",
        "img58.png",
        "img59.png",
        "img60.png",
        "img61.png",
        "img62.png",
        "img63.png",
        "img64.png",
    ]

    selected_images = select_even_slices(img_paths, top_k=10)
    print(selected_images)


# def top_time_series(df, top=10, time_seq=True, del_deficiency=True, invalid_cutoff=0.5):
#     """Filter and sort the dataframe to keep the top `n` time series per patient based on positive scores.

#     Args:
#         df (pd.DataFrame): DataFrame containing image scores and patient information.
#         top (int): Number of top records to keep per patient.
#         time_seq (bool): Whether to keep the records in time sequence order.
#         del_deficiency (bool): Whether to delete patients with less than `top` records.
#         invalid_cutoff (float): Cutoff for invalid scores.

#     Returns:
#         pd.DataFrame: Filtered and sorted DataFrame.
#     """
#     # Filter out invalid scores
#     df = df[df["Invalid_score"] <= invalid_cutoff]
#     # Sort by patient and positive score
#     df.sort_values(["Patient", "Pos_score"], ascending=[1, 0], inplace=True)

#     if time_seq:
#         df = df.groupby(["Patient"]).head(top).sort_index()
#     else:
#         df = df.groupby(["Patient"]).head(top)

#     group_size = df.groupby("Patient").size()
#     del_group = group_size[group_size < top]

#     if del_deficiency:
#         df = df[~df["Patient"].isin(del_group.index)]

#     return df


# def X_fromdf(df_top):
#     """Read images from a DataFrame and prepare them as input tensors for a PyTorch model.

#     Args:
#         df_top (pd.DataFrame): DataFrame containing file paths to the images.

#     Returns:
#         torch.Tensor: Tensor containing the preprocessed images.
#     """
#     images = [read_ct_img_bydir(file).unsqueeze(0) for file in df_top['File'].tolist()]
#     X = torch.stack(images)
#     X = X.permute(0, 2, 3, 1).unsqueeze(0)  # (batch, channels, height, width)
#     return X


# def X_y_patient_fromdf(df_top):
#     """Prepare the dataset for training by reading images and labels from the filtered DataFrame.

#     Args:
#         df_top (pd.DataFrame): Filtered DataFrame containing file paths and labels.

#     Returns:
#         tuple: (patient_list, X, y)
#             - patient_list (list): List of patient IDs.
#             - X (torch.Tensor): Feature matrix containing images.
#             - y (list): List of labels.
#     """
#     X = []
#     y = []
#     patient_list = []

#     for patient, ds in tqdm(df_top.groupby('Patient')):
#         X_patient = torch.stack([read_ct_img_bydir(file) for file in ds['File'].tolist()])
#         X_patient = X_patient.unsqueeze(1).permute(1, 2, 3, 0)  # Add channel dimension and transpose
#         # X_patient=np.array([read_ct_img_bydir(file).toarray() for file in ds['File'].tolist()])
#         # X_patient=X_patient[:,:,:,np.newaxis].transpose(3,1,2,0)
#         X.append(X_patient)
#         y.append(ds['Type'].iloc[0])
#         patient_list.append(patient)

#     X = torch.cat(X, dim=0).permute(3, 0, 1, 2)  # Concatenate and transpose back
#     return patient_list, X, y # check changes from original
