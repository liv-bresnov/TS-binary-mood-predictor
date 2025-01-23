# TaylorSwift song mood predictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import chi2_contingency


# Initialize lists for features and labels
features = []
labels = []

# Read song names from file
with open("3 ugers/song_list.txt") as file:
    song_list = [line.rstrip() for line in file]

# Iterate over each song in the list
for filename in song_list:
    
    # Load dictionary of features
    with open("feature_dict.pkl", "rb") as f:
        feature_dict = pickle.load(f)

    # Load dictionary of labels
    with open("pos_neg_dict.pkl", "rb") as f:
        pos_neg_dict = pickle.load(f)

    # Append features and label to the list
    features.append(feature_dict[filename])
    labels.append(pos_neg_dict[filename])



# Convert the feature list and labels to PyTorch tensors
X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Normalize the data using StandardScaler (convert to numpy first for scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.numpy())
X_test = scaler.transform(X_test.numpy())

# Convert back to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Create DataLoader for Batching the dataset, reduce comp load per step
train_dataset = TensorDataset(X_train, y_train.view(-1, 1))
test_dataset = TensorDataset(X_test, y_test.view(-1, 1))

batch_size = 5
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the model using Sequential
model = nn.Sequential(
    nn.Linear(11, 16),  # Input layer: 11 features -> 16 neurons
    nn.ReLU(),         # Activation function
    nn.Dropout(0.55),   # Dropout with 55%
    nn.Linear(16, 8),  # Hidden layer: 16 neurons -> 8 neurons
    nn.ReLU(),         # Activation function
    nn.Dropout(0.55),   # Dropout with 55%
    nn.Linear(8, 1),   # Output layer: 8 neurons -> 1 neuron
    nn.Sigmoid()       # Sigmoid activation for binary classification
)

# Loss function
loss_fn = torch.nn.BCELoss() # Chose binary cross-entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Training loop
epochs = 100
train_loss = []
test_loss = []

for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward() # Backpropagation
        optimizer.step() # Updates weights based on graidents from backprop
        epoch_train_loss += loss.item()

    train_loss.append(epoch_train_loss / len(train_loader)) # Mean traning loss

    # Testing phase
    model.eval()
    epoch_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            epoch_test_loss += loss.item()

    test_loss.append(epoch_test_loss / len(test_loader)) # Mean test loss

    # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}")

# Evaluating accuracy on test set
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    y_pred_labels = (y_pred_test >= 0.5).float()
    test_accuracy = accuracy_score(y_test.numpy(), y_pred_labels.numpy())

print(f"testaccuracy: {test_accuracy:.4f}")


# Training and test loss
"""
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Training Loss", color='royalblue')
plt.plot(test_loss, label="Test Loss", color='lightskyblue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid(True)
plt.show()
"""

# Dropout
"""
plt.figure(figsize=(10, 6))
plt.plot(dropout_values, train_losses, label='Train Loss', color='royalblue')
plt.plot(dropout_values, test_losses, label='Test Loss', color='lightskyblue')
plt.title('Train and Test Loss vs Dropout Values')
plt.xlabel('Dropout Values')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
"""


# Confusion matrix
"""
# Calculate confusion matrix
cm = confusion_matrix(y_test.numpy(), y_pred_labels.numpy())
print("Confusion Matrix:")
print(cm)

# For sexy visual
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
"""