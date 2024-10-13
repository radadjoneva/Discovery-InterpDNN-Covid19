import torch


class CTCFMultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, ct_dataset, cf_dataset, split="train"):
        self.ct_dataset = ct_dataset
        self.cf_dataset = cf_dataset
        self.classes = ct_dataset.classes
        self.split = split

    def __len__(self):
        assert len(self.ct_dataset) == len(self.cf_dataset), "Dataset sizes must match!"
        return len(self.ct_dataset)

    def __getitem__(self, idx):
        # Get data from both datasets
        ct_data = self.ct_dataset[idx]
        cf_data = self.cf_dataset[idx]

        # Patient IDs and labels must match in both datasets
        assert ct_data["patient_id"] == cf_data["patient_id"], "Patient ID mismatch!"
        assert torch.equal(ct_data["label"], cf_data["label"]), "Label mismatch!"

        # Create a combined data dictionary
        data = {
            "input": (cf_data["input"], ct_data["input"]),
            "label": ct_data["label"],
            "patient_id": ct_data["patient_id"],
            "kimg_paths": ct_data["kimg_paths"],
        }
        return data

    def class_distribution(self):
        return self.ct_dataset.class_distribution()
