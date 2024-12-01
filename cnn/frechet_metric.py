import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from cnn_utils import sequencesToDrawings
import pandas as pd

def main():

	print("Loading data")
	df_sample = pd.read_csv('./data_files/data.csv')
	df_generated = pd.read_csv('./data_files/generated_data.csv')
	class_name = "apple"

	print("Generating Drawings")
	sample_imgs = np.array(sequencesToDrawings(df_sample[df_sample.iloc[:, 0].isin([class_name])].iloc[:, 1].tolist()))
	generated_imgs = np.array(sequencesToDrawings(df_generated[df_generated.iloc[:, 0].isin([class_name])].iloc[:, 1].tolist(), generated=True))

	# Plot example data
	# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
	# axs[0].imshow(sample_imgs[10])
	# axs[0].set_title("Sample Image")
	# axs[0].axis("off")
	# axs[1].imshow(generated_imgs[10])
	# axs[1].set_title("Generated Image")
	# axs[1].axis("off")
	# plt.tight_layout()
	# plt.show()

	print("Creating Tensors")
	# Convert grayscale images to RGB by stacking the single channel three times
	sample_imgs_rgb = np.stack((sample_imgs,)*3, axis=-1)  # Shape: [N, H, W, 3]
	generated_imgs_rgb = np.stack((generated_imgs,)*3, axis=-1)  # Shape: [N, H, W, 3]

	# Convert to tensors and permute dimensions to [N, C, H, W]
	real_images_tensor = torch.tensor(sample_imgs_rgb, dtype=torch.uint8).permute(0, 3, 1, 2)
	generated_images_tensor = torch.tensor(generated_imgs_rgb, dtype=torch.uint8).permute(0, 3, 1, 2)

	print("Creating Datasets")
	real_dataset = TensorDataset(real_images_tensor)
	generated_dataset = TensorDataset(generated_images_tensor)

	batch_size = 64
	real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
	generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	fid = FrechetInceptionDistance(feature=768).to(device)

	print("Calculating FID Score")
	for real_batch in real_loader:
		fid.update(real_batch[0].to(device), real=True)
	for generated_batch in generated_loader:
		fid.update(generated_batch[0].to(device), real=False)

	fid_score = fid.compute()
	print(f"FID Score: {fid_score}")

if __name__ == "__main__":
	main()