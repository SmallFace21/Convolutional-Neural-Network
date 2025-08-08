# CIFAR-10 CNN Classification with Keras

### Course: CAP 4613 – Introduction to Deep Learning  
### Author: Nathan LaBar  
### Institution: Florida Atlantic University  
### Semester: Summer 2025  
### Assignment: 12  

---

##  Project Description

This project implements a convolutional neural network (CNN) using Keras to classify images in the **CIFAR-10** dataset — a standard benchmark dataset consisting of 60,000 32x32 color images in 10 classes.

The assignment explores the effects of:
- A baseline CNN without augmentation
- A CNN trained using data augmentation
- A CNN trained with batch normalization

It follows the CNN architecture outlined in the assignment prompt and includes multiple training setups to analyze performance, overfitting, and generalization.

---

## Files

- `Nathan_LaBar_Assignment12.ipynb`: Main Jupyter notebook containing:
  - Model definitions
  - Data preparation and preprocessing
  - Training routines for all 3 setups
  - Plotting of training/validation loss
  - Final evaluation and discussion

---

## Assignment Tasks

### Baseline CNN

- Two convolutional blocks (each with two `Conv2D` layers + `MaxPooling2D`)
- Flatten layer → Dense (512) → Output (10 softmax units)
- Activation: ReLU (after conv layers and FC layer)
- Loss: Categorical crossentropy
- Optimizer: Adam (lr = 0.001)
- Batch size: 32  
- Epochs: 50  
- `ModelCheckpoint` used to save best model based on **lowest validation loss**

---

### CNN with Data Augmentation

Same architecture as baseline. Additionally:

- Real-time data augmentation via `ImageDataGenerator`:
  - `rotation_range=10`
  - `width_shift_range=0.1`
  - `height_shift_range=0.1`
  - `horizontal_flip=True`

---

### CNN with Batch Normalization

- Adds `BatchNormalization` **after each Conv and FC layer but before activations**
- Bias disabled in conv and FC layers
- Optimizer: Adam (lr = 0.01)
- Batch size: 64
- Other training settings same as previous models

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages:
  ```bash
  pip install numpy matplotlib tensorflow keras
  ```

### Running the Notebook

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/cifar10-cnn.git
   cd cifar10-cnn
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook Nathan_LaBar_Assignment12.ipynb
   ```

3. Execute all cells to train and evaluate the models.

---

## Results & Observations

- **Training/Validation Loss Plots** are generated for each model.
- **Accuracy and Loss** are evaluated using the `evaluate()` method.
- Comparison of validation performance shows how:
  - Data augmentation improves generalization.
  - Batch normalization accelerates convergence and stabilizes training.
  - Higher learning rate for batch-norm model requires careful tuning.

---

## Discussion

The notebook concludes with answers to:
- Whether models are overfitting or underfitting.
- The impact of data augmentation and normalization on training and validation losses.
- Key takeaways on CNN optimization in deep learning workflows.

---

## License

This project is for educational purposes only as part of CAP 4613 at FAU.

---

## Acknowledgments

- **FAU Department of Computer Science and Engineering**
- TensorFlow & Keras documentation
- CIFAR-10 dataset by Alex Krizhevsky et al.
