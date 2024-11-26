from cnn_utils import sequencesToDrawings, onehotClasses
from model import CNNModel

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CNNTrainer():
	def __init__(self):
		self.classes = None
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = CNNModel().to(self.device)

	def train(self):

		df = pd.read_csv('./data_files/data.csv')
		self.classes = df.iloc[:, 0].unique().tolist()
		validateClassData = []
		validateImageData = []
		trainClassData = []
		trainImageData = []
		for item in self.classes:
			images = sequencesToDrawings(df[df.iloc[:, 0].isin([item])].iloc[:, 1].tolist())
			encodings = onehotClasses(df[df.iloc[:, 0].isin([item])].iloc[:, 0].tolist())
			validateClassData += encodings[:450]
			validateImageData += images[:450]
			trainClassData += encodings[450:]
			trainImageData += images[450:]

		validateClassData = np.array(validateClassData)
		validateImageData = np.array(validateImageData)
		trainClassData = np.array(trainClassData)
		trainImageData = np.array(trainImageData)
		
		# Convert data to PyTorch tensors
		train_images_tensor = torch.tensor(trainImageData, dtype=torch.float32).unsqueeze(1)
		train_labels_tensor = torch.tensor(np.argmax(trainClassData, axis=1), dtype=torch.long)
		validate_images_tensor = torch.tensor(validateImageData, dtype=torch.float32).unsqueeze(1)
		validate_labels_tensor = torch.tensor(np.argmax(validateClassData, axis=1), dtype=torch.long)

		train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
		validate_dataset = TensorDataset(validate_images_tensor, validate_labels_tensor)

		train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
		validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)

		# Define the model, criterion, and optimizer
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(self.model.parameters(), lr=0.001)

		# Training loop
		epochs = 10
		for epoch in tqdm(range(epochs), desc="Epoch:"):
			self.model.train()
			running_loss = 0.0
			for images, labels in train_loader:
				images, labels = images.to(self.device), labels.to(self.device)

				outputs = self.model(images)  # No softmax during training
				loss = criterion(outputs, labels)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				running_loss += loss.item()

			# Validation
			self.model.eval()
			val_loss = 0.0
			correct = 0
			total = 0
			with torch.no_grad():
				for images, labels in validate_loader:
					images, labels = images.to(self.device), labels.to(self.device)

					outputs = self.model(images, applySoftmax=True)  # Apply softmax for probabilities
					loss = criterion(outputs, labels)
					val_loss += loss.item()

					_, predicted = torch.max(outputs, 1)
					correct += (predicted == labels).sum().item()
					total += labels.size(0)

			print(f"Epoch [{epoch + 1}/{epochs}], "
				f"Training Loss: {running_loss / len(train_loader):.4f}, "
				f"Validation Loss: {val_loss / len(validate_loader):.4f}, "
				f"Validation Accuracy: {100 * correct / total:.2f}%")

		torch.save(self.model.state_dict(), 'model-state.pt')

	def test(self):
		self.model.load_state_dict(torch.load('model-state.pt', weights_only=True))
		self.model.eval()
		
		df = pd.read_csv('./data_files/test_data.csv')
		c = df.iloc[:, 0].tolist()
		s = df.iloc[:, 1].tolist()
		drawings = np.array(sequencesToDrawings(s))

		image_tensor = torch.tensor(drawings, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
		image_tensor = image_tensor.squeeze(2).to(self.device)

		output_vectors = []

		# Process all drawings
		for i, drawing in enumerate(drawings):
			# Convert the drawing into a tensor
			image_tensor = torch.tensor(drawing, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Normalize
			image_tensor = image_tensor.squeeze(2).to(self.device)  # Remove unnecessary dimensions

			# Get the model's output
			with torch.no_grad():
				output_vector = self.model(image_tensor, applySoftmax=True).cpu().numpy().flatten()
			
			output_vectors.append(output_vector)

			# Only plot the first image
			if i == 0:
				with open('./data_files/data_index.json', 'r') as f:
					indexes = json.load(f)
				# Display the first image with the output vector in the legend
				fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size to leave room for the text
				ax.imshow(drawing, cmap='gray')

				# Format the output vector
				output_str = ''.join([f'{v}: {output_vector[i]:.2f}\n' for i, v in enumerate(indexes.keys())])

				# Position the text to the right of the image
				plt.text(1.05, 0.5, f'{output_str}', transform=ax.transAxes, fontsize=12, verticalalignment='center')

				# Set title
				plt.title("Model Prediction Example")

				# Adjust layout to prevent clipping
				plt.subplots_adjust(right=0.75)  # Adjust this to leave enough space for the text

				# Show the plot
				plt.show()

def main():
	trainer = CNNTrainer()
	x = input("Train (1)\nTest(2)\n: ")
	if x == "1":
		trainer.train()
	elif x == "2":
		trainer.test()

if __name__ == "__main__":
	main()