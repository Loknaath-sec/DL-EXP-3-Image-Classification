# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion MNIST dataset. The Fashion MNIST dataset contains images of clothing items such as T-shirts, trousers, dresses, and shoes, and the model aims to classify them into their respective categories correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/c785e5f3-9533-4bf5-8ae0-68ecc38aa273" />


## DESIGN STEPS

#### STEP 1: Problem Statement
Define the objective of classifying fashion items (such as shirts, shoes, bags, etc.) using a Convolutional Neural Network (CNN).

#### STEP 2:Dataset Collection
Use the Fashion MNIST dataset, which contains 60,000 training images and 10,000 test images of labeled clothing and accessory items.

#### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

#### STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers tailored for 10 fashion categories.

#### STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

#### STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

#### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM
#### Name: LOKNAATH P
#### Register Number: 212223240080

```python
class CNNClassifier(nn.Module):
  def __init__(self):
    super(CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128) # Adjusted input features for fc1
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size(0), -1) # Flatten the image
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: Loknaath P')
        print('Register Number: 212223240080')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="412" height="218" alt="image" src="https://github.com/user-attachments/assets/830c1168-e136-4c87-9300-48c469dc7400" />


### Confusion Matrix

<img width="811" height="698" alt="image" src="https://github.com/user-attachments/assets/1320fb90-93a5-4fec-b52b-da629f4a1bb6" />


### Classification Report

<img width="593" height="373" alt="image" src="https://github.com/user-attachments/assets/e556a9cd-b8ef-4920-b960-1ae3bf758495" />



### New Sample Data Prediction

<img width="659" height="641" alt="image" src="https://github.com/user-attachments/assets/00d6c4b0-cba7-4826-badb-418f7a363c37" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
