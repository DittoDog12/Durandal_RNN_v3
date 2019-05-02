from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os, re, random, unidecode, time, collections
#from tensorflow import keras
import Tensor, Utilities

#https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/
#https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms
#https://www.tensorflow.org/tutorials/sequences/text_generation


# General settings
# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable Eager Execution
# This makes the program run commands immediatly rather than having to create and then run a session
tf.enable_eager_execution()

# Command Line Arguments  
import argparse
parser = argparse.ArgumentParser()    
parser.add_argument('-i','--input', default='Input.txt', help='Data Source')
parser.add_argument('--seq_length', default=200, type=int, help='Input Sequence Length, Default 200')
parser.add_argument('--rnn_size', default=1024, type=int, help='Neural Network Size, Default 1024')
parser.add_argument('-e', '--epochs', default=5, type=int, help='Number of Training Epochs, Default 5')
parser.add_argument('--buffer', default=100, type=int, help="Buffer size to shuffle training data between batches, Default 100")
parser.add_argument('--batch_size', default=24, type=int, help="Size of training batches, Default 24")
parser.add_argument('--embedding_dim', default=256, type=int, help="Set Embedding Dimension")
parser.add_argument('--resume_last', action='store_true', help="Specify to open last checkpoint")
parser.add_argument('--checkpoint_every', default=10, type=int, help="Set how often to save a checkpoint, default 10")
parser.add_argument('-se', '--skip_evaluation', action='store_true', help="Specify to skip Evaluation")
parser.add_argument('-st', '--skip_train', action='store_true', help="Specify to skip training")
parser.add_argument('--seed', default='nul', help="Seed for training data, if unused will open Data\\Seed.txt")
parser.add_argument('--num_generate', default=200, type=int, help='Number of characters to predict, Default 200')
parser.add_argument('--temperature', default=1, type=float, help='Evaluation Temp')

args = parser.parse_args()
training_file = os.path.join('.\\Data\\', args.input) # string path of training file, first join the path and the argument
seq_length = args.seq_length # int Input Sequence Length 
units = args.rnn_size # int Number of RNN units inside the LSTM layer
EPOCHS = args.epochs # int Number of Training Epochs to run
Buffer_Size = args.buffer # int amount to shuffle each batch by
Batch_Size = args.batch_size # int Size of each training batch
embedding_dim = args.embedding_dim # int Size of Embedding (Input) Layer
UseLast = args.resume_last # bool Control if training session reopens the weights from the last session
CheckPointEvery = args.checkpoint_every # int Interval to save checkpoint files
DontEval = args.skip_evaluation # bool Control if Evaluation step should be performed
DontTrain = args.skip_train # bool Control if Training step should be performed
Seed = args.seed # string Evaluation step initialization seed
num_generate = args.num_generate # int Number of characters Evaluation step should output
Temperature = args.temperature # float Evaluation Temperature, alters the randomness of the output
                                # Higher Temperature produces more surprising results
                                # Lower Temperature produces more predicable results

Training_batch_size = Batch_Size # int Make a note of the batch size for outputting to the file for logging purposes, batch size gets overwritten at the evaluation stage


# Debug options
#DontTrain = True
#Seed = 'slivers'
#UseLast = False

# Setup output file
OutputFilename = time.strftime("%Y%m%d %H%M%S") + ".txt"
OutputFile = os.path.join('.\\Output', OutputFilename)

# Load data
file = open(training_file, 'r')
text = file.read()
file.close

# Set up storage path for Checkpoint and dictionary files
ckptdir = ".\\Checkpoints"

# Calculate number of block types
vocab = sorted(set(text))

# Create indices for each unique character (Create dictionary)
char2idx = {u:i for i, u in enumerate(vocab)}
# Create reverse lookup
idx2char = np.array(vocab)

# Calculate number of unique characters, print result
vocab_size = len(vocab)
print('Vocab Size: %d' % vocab_size)

# Load Utilities
utils = Utilities.main()

# Optional: output dictionary to file for testing
utils.OutputDictionary(vocab)

# Encode the training text
training_data = utils.Encode(text, char2idx)
# Create Training Examples/Targets
dataset = utils.CreateDatasets(training_data, seq_length, idx2char)

# Create Training batches
dataset = dataset.shuffle(Buffer_Size).batch(Batch_Size, drop_remainder=True)

# Load tensor
tensor = Tensor.main()

# BEGIN MAIN TRAINING ROUTINE
# If DontTrain = false, Train as normal
# Set true by -st arg
if DontTrain == False:
    # Load model
    model = tensor.create_model(units, vocab_size, Batch_Size, embedding_dim)

    # Open the last weights if specified
    if UseLast == True:
        model.load_weights(tf.train.latest_checkpoint(ckptdir))
        #loss, acc = model.evaluate(dataset.repeat())
        #print ("Restored Model, accuracy: {:5.2f}%".format(100*acc))
        #model.build(tf.TensorShape([1, None]))
    
    examples_per_epoch = seq_length
    steps_per_epoch = Batch_Size

    # Create an optimizer function
    optimizer = tf.train.AdamOptimizer()

    # Train
    # Simple train method, not much room for control
    history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[utils.Checkpointer(ckptdir, CheckPointEvery)])
    checkpint_prefix = os.path.join(ckptdir, "ckpt_{epoch}")
    model.save_weights(checkpint_prefix.format(epoch=EPOCHS))
    
    # Advanced train method, allows for more control
    """for epoch in range(EPOCHS):
        start = time.time()

        # Initializing the hidden state at the start of each epoch
        hidden = model.reset_states()

        for (batch_n, (inp, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                # Feed the hidden state back into the model
                predictions = model(inp)
                loss = tf.losses.sparse_softmax_cross_entropy(target, predictions)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Print batch loss after each batch
            if batch_n % 100 == 0:
                template = 'Epoch {} Batch {} Loss {:.4f}'
                print (template.format(epoch + 1, batch_n, loss))

        # Saving checkpoint according to argument
        if (epoch + 1) % CheckPointEvery == 0:
            model.save_weights(ckptdir.format(epoch=epoch))

        # Print epoch loss after each epoch
        print ('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
        print ('Time taken for 1 Epoch {} sec\n'.format(time.time() - start))
    
    # Save final checkpoint
    model.save_weights(ckptdir.format(epoch=epoch)) """


# BEGIN MAIN PREDICTION ROUTINE
# If DontEval = false, Predict as normal
# Set true by -se arg
if DontEval == False:
    # Load seed
    # If Seed = 'nul' open and parse the data\seed.txt file
    # Else load the Seed arg straight to the predictor
    if Seed == "nul":
        file = open(".\\Data\\Seed.txt", 'r')
        seed_text = file.read()
        file.close
    else:
        seed_text = Seed

    # Restore latest checkpoint
    # We need to keep the predictions simple so override the batch size to 1
    # however we need to build a new model and load the previous weights into the new model
    # Models can only run a the batch size they were built with
    Batch_Size = 1
    model = tensor.create_model(units, vocab_size, Batch_Size, embedding_dim)
    model.load_weights(tf.train.latest_checkpoint(ckptdir))
    model.build(tf.TensorShape([1, None]))
    print ("Restored Model summary: ")
    model.summary()

    # Encode Seed and convert to array
    input_eval = [char2idx[s] for s in seed_text]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store output
    text_generated = ''

    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        # Remove the batch dimensions
        predictions = tf.squeeze(predictions, 0)

        # Use Multinominal distribution to predict the next character
        predictions = predictions / Temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character back to the model along with the previous hidden state
        # Recurrent part
        input_eval = tf.expand_dims([predicted_id], 0)

        # Decode the predicted character and append to the text generated
        text_generated += idx2char[predicted_id]

    # Save text generated to file
    # First output the settings used
    # Then save the complete output
    f = open(OutputFile, 'w+')
    print ("Skipped Training: {0}, Epochs: {1}, Sequence Length: {2}, Batch Size: {3}, Number Generate: {4}, Seed: {5}, Resumed Training: {6}, RNN Size: {7}, Temperature {8}".format(DontTrain, EPOCHS, seq_length, Training_batch_size, num_generate, seed_text, UseLast, units, Temperature), file=f)
    print (text_generated, file=f)
    print ("Saved to {0}".format(OutputFile))
    f.close()