import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_latent_feature_distributions(z, labels, title, ax=None):
    """
    Plots the distributions of latent features using separate violin plots for general and class-specific distributions.
    
    Args:
        z (torch.Tensor): The latent representations.
        labels (np.array): The class labels corresponding to each z vector.
        title (str): The title of the plot.
        ax (matplotlib axis, optional): An axis to plot on. If None, a new figure is created.
    """
    z = z.detach().cpu().numpy()  # Convert to numpy
    num_features = z.shape[1]

    ix_to_class = {0: "Control", 1: "Type I", 2: "Type II"}
    class_palette = {"Control": "royalblue", "Type I": "darkorange", "Type II": "forestgreen"}
    
    # Convert to DataFrame for easier plotting with seaborn
    df = pd.DataFrame(z, columns=[f'{i+1}' for i in range(num_features)])
    df['Class'] = labels  # Add class labels
    df['Class'] = df['Class'].map(ix_to_class)  # Map class indices to class names
    df_melted = df.melt(id_vars='Class', var_name='Latent Feature', value_name='Value')
    df_melted['Class'] = pd.Categorical(df_melted['Class'], categories=["Control", "Type I", "Type II"], ordered=True)

    # Determine global y-limits (min/max) across all features
    abs_max = max(abs(df_melted['Value'].min()), abs(df_melted['Value'].max()))
    y_min = -abs_max
    y_max = abs_max
    
    # General violin plot across all classes
    if ax is None:  # If no axis is provided, create a new figure
        plt.figure(figsize=(14, 7))
        sns.violinplot(x='Latent Feature', y='Value', data=df_melted, inner='quartile', color='palevioletred', density_norm="count", cut=0)
        plt.title(f'{title} - General Distribution', fontsize=14)
        plt.xlabel('Latent Feature', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        save_path = f"research/case_study/biomed/results/interpretability/latent_space/z_distributions/{title}_general.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        sns.violinplot(x='Latent Feature', y='Value', data=df_melted, inner='quartile', color='palevioletred', density_norm="count", cut=0, ax=ax)
        ax.set_title(f'{title} - General Distribution', fontsize=14)
        ax.set_xlabel('Latent Feature', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True)

    # Class-specific violin plots (no need for axis here)
    num_cols = 4  # Number of columns for the subplot grid
    num_rows = (num_features + num_cols - 1) // num_cols  # Number of rows needed based on num_features
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 3))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, feature in enumerate(df_melted['Latent Feature'].unique()):
        ax = axes[i]
        sns.violinplot(x='Class', y='Value', hue='Class', data=df_melted[df_melted['Latent Feature'] == feature],
                       ax=ax, palette=class_palette, inner='quartile', legend=False, density_norm="count", hue_order=["Control", "Type I", "Type II"], cut=0)
        ax.set_title(f'Feature {i+1}', fontsize=10)
        ax.set_xlabel('') 
        ax.set_ylabel('')
        ax.set_ylim(y_min, y_max)
        ax.grid(True)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle(f'{title} - Class Distributions', fontsize=16)

    # Save the figure
    save_path = f"research/case_study/biomed/results/interpretability/latent_space/z_distributions/{title}_feature_class_distribution.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()



def plot_latent_space(z_real, z_sampled, labels_real, labels_pred, labels_sampled, title, classes, sampled_label):
    dim = z_real.shape[1]  # Determine if it's 2D or 3D
    
    if dim == 2:
        plt.figure(figsize=(10, 5))
        unique_labels = np.unique(labels_real)
        cmap = plt.get_cmap('tab10')
        
        # Plot real data with different colors for each class
        for i, label in enumerate(unique_labels):
            color = cmap(i % 10)  # Cycle through the colors if more than 10 classes
            plt.scatter(z_real[labels_real == label, 0], z_real[labels_real == label, 1], 
                        color=color, marker='o', label=f'z Class {classes[i]}', alpha=0.6)
        
            # Mark incorrectly classified points
            incorrect = labels_pred[labels_real == label] != labels_real[labels_real == label]
            for idx in np.where(incorrect)[0]:
                # Plot the smaller black dot
                plt.scatter(z_real[labels_real == label][idx, 0], z_real[labels_real == label][idx, 1], 
                            color='black', marker='.', alpha=1.0, s=30)
                
                # Plot the cross with the color of the predicted class
                pred_label = labels_pred[labels_real == label][idx]
                pred_color = cmap(pred_label % 10)
                plt.scatter(z_real[labels_real == label][idx, 0], z_real[labels_real == label][idx, 1], 
                            color=pred_color, marker='.', alpha=0.8, s=15)
        
        # Plot sampled data with predicted labels
        unique_sampled_labels = np.unique(labels_sampled)
        for i, label in enumerate(unique_sampled_labels):
            plt.scatter(z_sampled[labels_sampled == label, 0], z_sampled[labels_sampled == label, 1], 
                        color=cmap(i % 10), marker='x', label=f'{sampled_label} Class pred {classes[i]}', alpha=0.6)
        
        plt.title(title)
        plt.legend()
        save_path = f"research/case_study/biomed/results/interpretability/latent_space/dim_reduction/{title}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        plt.close()
    
    elif dim == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = np.unique(labels_real)
        cmap = plt.get_cmap('tab10')
        
        # Plot real data with different colors for each class
        for i, label in enumerate(unique_labels):
            color = cmap(i % 10)  # Cycle through the colors if more than 10 classes
            ax.scatter(z_real[labels_real == label, 0], z_real[labels_real == label, 1], z_real[labels_real == label, 2], 
                       color=color, marker='o', label=f'z Class {classes[i]}', alpha=0.6)
        
            # Mark incorrectly classified points
            incorrect = labels_pred[labels_real == label] != labels_real[labels_real == label]
            for idx in np.where(incorrect)[0]:
                # Plot the smaller black dot
                ax.scatter(z_real[labels_real == label][idx, 0], z_real[labels_real == label][idx, 1], z_real[labels_real == label][idx, 2], 
                           color='black', marker='.', alpha=1.0, s=30)
                
                # Plot the cross with the color of the predicted class
                pred_label = labels_pred[labels_real == label][idx]
                pred_color = cmap(pred_label % 10)
                ax.scatter(z_real[labels_real == label][idx, 0], z_real[labels_real == label][idx, 1], z_real[labels_real == label][idx, 2], 
                           color=pred_color, marker='.', alpha=0.8, s=15)
        
        # Plot sampled data with predicted labels
        unique_sampled_labels = np.unique(labels_sampled)
        for i, label in enumerate(unique_sampled_labels):
            ax.scatter(z_sampled[labels_sampled == label, 0], z_sampled[labels_sampled == label, 1], z_sampled[labels_sampled == label, 2], 
                       color=cmap(i % 10), marker='x', label=f'{sampled_label} Class pred {classes[i]}', alpha=0.6)
        
        ax.set_title(title)
        ax.legend()
        save_path = f"research/case_study/biomed/results/interpretability/latent_space/dim_reduction/{title}_3D.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        plt.close()