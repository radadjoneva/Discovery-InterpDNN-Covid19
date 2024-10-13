import torch


class ActivationsRecorder:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.patient_id = None  # Store current patient ID
        self.split = None  # Store current split
        self.outcome = None # Store current outcome

    def get_activation(self, name):
        def hook(module, input, output):
            # Store the activations
            if self.patient_id not in self.activations:
                self.activations[self.patient_id] = {}
                self.activations[self.patient_id]["split"] = self.split
                self.activations[self.patient_id]["outcome"] = self.outcome
            self.activations[self.patient_id][name] = output.detach().cpu()
        
        return hook

    def record_activations(self, X_data, idx_df):
        # Record activations for each data point in the dataset
        for index, row in X_data.iterrows():
            self.patient_id = idx_df.iloc[index]["Patient ID"]
            self.split = idx_df.iloc[index]["Split"]
            self.outcome = idx_df.iloc[index]["Outcome"]
            input_tensor = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add batch dimension

            # Run the model on the input data point; hooks will automatically capture activations
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            self.activations[self.patient_id]["prediction"] = prediction