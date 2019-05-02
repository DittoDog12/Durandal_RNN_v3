import tensorflow as tf
import numpy as np
import os, re, random, unidecode, time
#from tensorflow import keras
#from keras import utils as ku
#from keras.preprocessing.sequence import pad_sequences
from pickle import dump, load

#https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
#https://www.tensorflow.org/guide/eager
class main:
    def Checkpointer(self, ckptdir, ckpt_Period):
        checkpint_prefix = os.path.join(ckptdir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = checkpint_prefix,
            save_weights_only=True,
            period=ckpt_Period
        )
        return checkpoint_callback


    def Encode(self, training_text, dictionary):
        # Create encoded training file from input file using dictionary
        training_data = np.array([dictionary[c] for c in training_text])
        return training_data

    def CreateDatasets(self, src_text, seq_length, dictionary):
        char_dataset = tf.data.Dataset.from_tensor_slices(src_text)
        sequence = char_dataset.batch(seq_length+1, drop_remainder=True)
        # Print example of batch
        """ print ("Batch Example: ")
        for item in sequence.take(5):
            print(repr(''.join(dictionary[item.numpy()]))) """
        return sequence.map(self.slicearray)

    def slicearray(self, chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    def OutputDictionary(self, unique):
        f = open(".\\Data\\Vocab.txt", 'w+')
        for char in unique:
            print (char, file=f)
        f.close()
