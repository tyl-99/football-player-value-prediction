# Football Player Value Prediction using Feedforward Neural Network (FFN)

This project demonstrates the use of a Feedforward Neural Network (FFN) to predict the market value of football players based on various features.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

## Overview
![image](https://github.com/tyl-99/football-player-value-prediction/assets/71328888/57b37e27-6a6d-4351-9c2f-0372bf26a0c9)

This project aims to predict the market value of football players using a Feedforward Neural Network (FFN). The model is trained on a dataset containing various features related to the players, such as age, position, performance statistics, and more.

## Dataset

The dataset used for this project includes features and market values of football players. It consists of multiple attributes such as age, nationality, club, position, and various performance metrics. The dataset is split into training and test sets for model evaluation.

## Model Architecture

The Feedforward Neural Network (FFN) used in this project consists of multiple fully connected layers with ReLU activation functions. The architecture is designed to learn complex relationships between the input features and the target variable (player market value).

Here is a summary of the architecture:

```python
class PlayerValueFFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PlayerValueFFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

## Training

The model is trained for several epochs using the Adam optimizer and Mean Squared Error (MSE) loss function. The training process involves optimizing the model to minimize the prediction error, improving its ability to predict player values accurately.

## Results

The model achieves good performance in predicting the market value of football players. The results include evaluation metrics such as Mean Squared Error (MSE) and R-squared (R²) score, indicating the model's accuracy and reliability.

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn

You can install the required packages using the following command:

```bash
pip install torch pandas numpy scikit-learn
```

## Usage

To run the notebook and train the model, use the following command:

```bash
jupyter notebook FootballPlayerValuePrediction.ipynb
```

You can also use the trained model to predict the market value of football players in your own dataset by following the steps in the notebook.

