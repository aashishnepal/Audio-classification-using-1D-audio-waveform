# UrbanSound8K Audio Classification

This project involves classifying environmental sounds from the UrbanSound8K(https://urbansounddataset.weebly.com/download-urbansound8k.html) dataset using a neural network. The model is trained to predict the type of sound from 8k+ audio clips, each less than 4 seconds in duration. The dataset is organized into 10 folds, and metadata provides the class labels for each audio file. The project includes data preprocessing, model training, evaluation, and prediction.

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Dataset
The UrbanSound8K dataset consists of 8732 labeled sound excerpts (<= 4s) of urban sounds from 10 classes:
- Air conditioner
- Car horn
- Children playing
- Dog bark
- Drilling
- Engine idling
- Gunshot
- Jackhammer
- Siren
- Street music

The dataset is organized into 10 folds to facilitate cross-validation.

## Requirements
- Python 3.x
- TensorFlow 2.x
- librosa
- pandas
- numpy
- scikit-learn

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Data Preprocessing
The preprocessing involves extracting MFCC (Mel Frequency Cepstral Coefficients) features from each audio file.

### Feature Extraction
```python
import pandas as pd
import os
import librosa
import numpy as np

audio_dataset_path = 'UrbanSound8K/audio/'
metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features
```

## Model Architecture
The model is a Sequential neural network with multiple Dense layers and uses LeakyReLU activation, BatchNormalization, and Dropout for regularization.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

model = Sequential()
model.add(Dense(256, input_shape=(40,), kernel_regularizer=l2(0.001)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, kernel_regularizer=l2(0.001)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, kernel_regularizer=l2(0.001)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_labels, kernel_regularizer=l2(0.001)))
model.add(Activation('softmax'))

optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
```

## Training
The model is trained with early stopping and model checkpointing to save the best model.

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.keras', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

num_epochs = 200
num_batch_size = 32

start = datetime.now()

history = model.fit(
    X_train, y_train,
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(X_test, y_test),
    callbacks=[checkpointer, early_stopping],
    verbose=1
)

duration = datetime.now() - start
print("Training completed in time: ", duration)
```

## Evaluation
Evaluate the model on the test set and print the accuracy.

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')
```

## Prediction
Use the saved model to predict the class of new audio samples.

### predictor.ipynb
```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('saved_models/audio_classification.keras')

def predict(file_name):
    features = features_extractor(file_name)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)
    return predicted_label

# Example usage
file_name = 'UrbanSound8K/audio/fold1/101415-3-0-2.wav'
print(predict(file_name))
```

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/urbansound8k-audio-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd urbansound8k-audio-classification
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the preprocessing and training scripts.
5. Use the `predictor.ipynb` notebook for prediction.

## Results
The trained model achieves an accuracy of approximately `85%` on the test set. Detailed training logs and accuracy graphs are provided in the `results` directory.

## Acknowledgements
- UrbanSound8K dataset: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
- TensorFlow and Keras for the deep learning framework
- librosa for audio processing

Feel free to open issues or submit pull requests to improve the project.