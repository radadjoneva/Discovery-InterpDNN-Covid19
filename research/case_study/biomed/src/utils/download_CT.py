import os
import zipfile

import wget


def download_CT_data(url, idx_list, output_dir):
    """Download CT data from the iCTCF website for all patients."""
    for i in idx_list:
        url_patient = url + str(i) + ".zip"

        if os.path.exists(os.path.join(output_dir, f"Patient%20{i}.zip")) or os.path.exists(
            os.path.join(output_dir, f"Patient {i}")
        ):
            print(f"Patient {i} already downloaded")
            continue

        try:
            print(f"\nDownloading patient {i}...")
            wget.download(url_patient, out=output_dir)
        except Exception as e:
            if "404: Not Found" in str(e):
                print(f"Error: Patient {i} Not Found")
            else:
                print(f"An error occurred while downloading patient {i}: {e}")


def download_CT_images(url, output_dir):
    """Download manually labelled CT images from the iCTCF website used for training the CT_images_CNN."""
    classes = ["NiCT", "pCT", "nCT"]
    for label in classes:
        url_class = url + label + ".zip"
        print(url_class)
        try:
            print(f"\nDownloading {label} images...")
            wget.download(url_class, out=output_dir)
        except Exception as e:
            print(f"An error occurred while downloading CT images: {e}")


def unzip_files(directory):
    """Unzip all .zip files in the specified directory."""
    for item in os.listdir(directory):
        if item.endswith(".zip"):
            file_path = os.path.join(directory, item)
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(directory)
            print(f"Extracted: {item}")
            os.remove(file_path)
            print(f"Deleted: {item}")


if __name__ == "__main__":
    url = "https://ngdc.cncb.ac.cn/ictcf/patient/CT/"
    # nb_patients = 1521
    patients_list = range(18, 100)
    output_dir = "research/case_study/biomed/datasets/iCTCF/CT/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    download_CT_data(url + "Patient%20", patients_list, output_dir)
    unzip_files(output_dir)

    # download_CT_images(url, output_dir)

    # wget.download("https://ngdc.cncb.ac.cn/ictcf/patient/CT/NiCT.zip", out=output_dir)
