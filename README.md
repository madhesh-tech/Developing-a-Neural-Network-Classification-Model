# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.
## Problem Statement
The problem is to develop a neural network classification model that can predict the correct customer segment (A, B, C, or D) for new customers based on given features. The model should learn from existing market data and accurately classify customers to support targeted marketing strategies.
## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="1013" height="683" alt="image" src="https://github.com/user-attachments/assets/67fed00d-b1fd-4340-8d18-5fccd4ccfc51" />

## DESIGN STEPS
### STEP 1: 
Data Collection and Understanding – Load the dataset, inspect features, and identify the target variable.


### STEP 2: 
Data Cleaning and Encoding – Handle missing values and convert categorical data and labels into numerical form.




### STEP 3: 
Feature Scaling and Data Splitting – Normalize features and split data into training and testing sets.




### STEP 4: 
Model Architecture Design – Define the neural network layers, neurons, and activation functions.




### STEP 5: 

Model Training and Optimization – Train the model using a loss function and optimizer through backpropagation.



### STEP 6: 


Model Evaluation and Prediction – Evaluate performance using metrics and make predictions on unseen data.

## Name : Madhesh I
## Reg No : 212224220055


## PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
data = pd.read_csv('/content/customers.csv')
data.head()

data.columns

# Drop ID column as it's not useful for classification
data = data.drop(columns=["ID"])

# Handle missing values
data.fillna({"Work_Experience": 0, "Family_Size": data["Family_Size"].median()}, inplace=True)

# Encode categorical variables
categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Encode target variable
label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])  # A, B, C, D -> 0, 1, 2, 3

# Split features and target
X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size): # Changed init to__init__
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4) # Changed fc3 to fc4 to avoid overwriting


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Initialize model
input_size = X_train.shape[1]
model = PeopleClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_model(model, train_loader, criterion, optimizer, epochs=100)

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name:Madhesh I")
print("Register No:212224220055")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Prediction for a sample input
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():https://github.com/madhesh-tech/Developing-a-Neural-Network-Classification-Model/tree/main
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name:Madhesh I")
print("Register No:212224220055)
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')

```
### Dataset Information
<img width="1241" height="257" alt="Screenshot 2026-02-06 114657" src="https://github.com/user-attachments/assets/29b0d9e2-2ad7-4590-9a5e-71866341e8ee" />

### OUTPUT

## Confusion Matrix

<img width="730" height="585" alt="Screenshot 2026-02-06 114538" src="https://github.com/user-attachments/assets/8d9b91b7-4228-4764-b8b1-4a980932d184" />

## Classification Report
<img width="584" height="431" alt="Screenshot 2026-02-06 114620" src="https://github.com/user-attachments/assets/8466cac5-63e8-4617-aff2-a84853f570c4" />

### New Sample Data Prediction
<img width="442" height="108" alt="Screenshot 2026-02-06 114601" src="https://github.com/user-attachments/assets/368645b4-4d72-48b2-ae58-51a4a2423224" />

## RESULT
Neural network classification model for the given dataset is successfully developed.

