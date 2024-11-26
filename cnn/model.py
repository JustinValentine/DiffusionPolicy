import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
	def __init__(self):
		super(CNNModel, self).__init__()

		self.num_classes = 25

		# Define the CNN architecture
		self.conv_layers = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1), # Single channel input
			nn.ReLU(),
			nn.MaxPool2d(2, 2),  # 128x128 to 64x64
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),  # 64x64 to 32x32
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)  # 32x32 to 16x16
		)

		self.fc_layers = nn.Sequential(
			nn.Flatten(),
			nn.Linear(128 * 16 * 16, 128),  # Adjust input dimensions
			nn.ReLU(),
			nn.Dropout(0.5),  # Optional: prevents overfitting
			nn.Linear(128, self.num_classes)  # Output layer with self.num_classes classes
		)

	def forward(self, x, applySoftmax=False):
		x = self.conv_layers(x)
		x = self.fc_layers(x)
		if applySoftmax:
			x = F.softmax(x, dim=1) # Softmax for probabilities when evaluating
		return x
