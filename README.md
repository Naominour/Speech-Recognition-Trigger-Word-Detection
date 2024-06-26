# Speech recognition - Trigger Word Detection

This project focuses on implementing a trigger word detection system using deep learning techniques. Trigger word detection, also known as wake word detection, is the process of identifying specific keywords or phrases in an audio stream to activate a voice assistant or initiate a specific action.


![Deep Learning](https://img.shields.io/badge/Skill-Deep%20Learning-yellow)
![Recurrent Neural Networks](https://img.shields.io/badge/Skill-Recurrent%20Neural%20Networks-blueviolet)
![TensorFlow](https://img.shields.io/badge/Skill-TensorFlow-orange)
![Keras](https://img.shields.io/badge/Skill-Keras-yellow)
![Speech Recognition](https://img.shields.io/badge/Skill-Speech%20Recognition-brightblue)
![LSTM](https://img.shields.io/badge/Skill-LSTM-brightblue)
![Python Programming](https://img.shields.io/badge/Skill-Python%20Programming-orange)

## Project Architecture

**Data Preprocessing:** Load and preprocess audio data, including generating spectrograms.
**Model Architecture:** Build a Recurrent Neural Network (RNN) with LSTM units to process audio sequences.
**Training:** Train the RNN model on labeled audio data to detect the trigger word.
**Evaluation:** Evaluate the model's performance on test data.
**Inference:** Deploy the model to detect trigger words in real-time audio streams.

## Frameworks and Libraries
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.3.3-red.svg?style=flat&logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-blue.svg?style=flat&logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.10.1-yellow.svg?style=flat&logo=SciPy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6.2-green.svg?style=flat&logo=matplotlib)

## Key Features

- Detects specific trigger words in audio streams.
- Utilizes LSTM-based RNN for sequence processing.
- Real-time detection capability.
- Flexible to adapt different trigger words and datasets.
- High accuracy and robustness in various audio environments.

## Usage
**Clone the repository:**
```bash
git clone https://github.com/yourusername/Trigger_Word_Detection.git
```
**Navigate to the project directory:**
```bash
cd Trigger_Word_Detection
```
**Install the required dependencies:**
```bash
pip install -r requirements.txt
```
**Run the Jupyter Notebook to see the implementation:**
```bash
jupyter notebook Trigger_word_detection_v2a.ipynb
```
## Implementation
The project uses a Long Short-Term Memory (LSTM) based Recurrent Neural Network (RNN) to process sequences of audio data. The model is trained on a dataset of audio clips labeled with trigger words. The network learns to recognize patterns associated with the trigger words and can then detect these words in new audio streams. The preprocessing steps involve converting audio signals into spectrograms, which are used as inputs to the neural network.

## Results

The trained model successfully detects trigger words in various audio clips. The performance metrics, such as accuracy and false positive rate, indicate that the model is effective and reliable for practical use.