# Durandal Recurrent Neural Network version 3
A simple character level LSTM designed for creating platformer levels based on a limited set of characters.

Based on information from the following blogs
https://www.tensorflow.org/tutorials/sequences/text_generation
https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/
https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms

Named after the AI Durandal in Bungie's Marathon video games.

Prerequisits:
tensorflow 1.13.1 
unidecode 1.0.23

OR For GPU support
tensorflow-gpu 1.13.1
unidecode 1.0.23
CUDA Toolkit 10
CUDNN for CUDA 10 v7.5.0.56

To use: Run Main.py
Default settings will open Data\Input.txt for training
Epochs 5, Batch size 24, Sequence Length 200, RNN Size 1024
All these opetiosn can be chagned, full list of arguments can be found with main.py -h