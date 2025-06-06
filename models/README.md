# Model Files

This directory should contain the trained model file for PneumoScan.

## Required Files

- `vgg16_model_new.keras`: The trained VGG16 model for pneumonia detection

## How to Add the Model

1. Place your trained model file (`vgg16_model_new.keras`) in this directory
2. Make sure the model file is named exactly `vgg16_model_new.keras`
3. The model should be in Keras format (`.keras` extension)

## Model Requirements

- Input shape: (224, 224, 3)
- Output: Binary classification (Normal/Pneumonia)
- Format: Keras model format

## Note

For security and size reasons, the model file is not included in the repository. You need to:
1. Train the model locally
2. Place the model file in this directory
3. Deploy the application with the model file

## Deployment Instructions

When deploying to Render:
1. Make sure to include the model file in your deployment
2. The model file should be in the `models` directory
3. The file name should be exactly `vgg16_model_new.keras` 