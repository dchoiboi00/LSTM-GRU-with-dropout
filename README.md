# Deep Learning

## RNN: LSTM and GRUs on Word-level Language Modelling

Based on the paper "Recurrent Neural Network Regularization" by Wojciech Zaremba et al. (2014), [link to paper](https://arxiv.org/abs/1409.2329).

This script implements and trains an RNN with LSTMs and GRUs using PyTorch and the Penn Tree Bank
dataset. The purpose of this task is to test how dropout improves the performance of LSTMs and GRUs in
recurrent networks.

## Prerequisites
```
Python 3.10+
PyTorch
NumPy
Matplotlib
WandB (Weights and Biases for plotting)
Google Colab (Optional: to run the script)
```
## Dataset
Penn Tree Bank Dataset is available to download at [here](https://paperswithcode.com/dataset/penn-treebank).

## Training + Logs

Our model builds a recurrent network that can be configured to use either LSTMs or GRUs. Dropout can also
be applied to improve generalization on the validation and test sets.

We have two functions run_experiments_no_dropout() and run_experiments_with_dropout() to train
our models and print the logs. You can adjust the hyperparameters (dropout rate, learning rate, learning rate
decay + scheduling) within the functions.

Example log:

```
batch no = 0 / 2323, avg train loss per word this batch = 5.235, words per second
= 3, lr = 1.176, since beginning = 3 mins,
batch no = 232 / 2323, avg train loss per word this batch = 4.402, words per
second = 606, lr = 1.176, since beginning = 3 mins,
...
batch no = 2088 / 2323, avg train loss per word this batch = 4.674, words per
second = 5161, lr = 1.176, since beginning = 3 mins,
batch no = 2320 / 2323, avg train loss per word this batch = 4.625, words per
second = 5699, lr = 1.176, since beginning = 3 mins,
Epoch 11: Start Learning Rate: 2.0, Dropout: 0.
Epoch 11: Train Loss: 4.
Epoch 11: Train Perplexity: 90.
Epoch 11: Validation Perplexity: 118.
Epoch 11: Test Perplexity: 114.
Saw better model at Epoch 11
```

We used WandB to plot our perplexities. After running one of the two functions, you should be able to click
on a link to view all previous runs. Our four best models in the writeup are prefixed with "Pass_", which you
should be able to filter from the search bar.

## Testing

At the end of the training process, the best model for each regularization technique is saved under
best_model_LSTM|GRU_LEARNING_RATE_DROUPOUT.pth in PATH_TO_MODEL directory, which can be loaded
later. To test the model with the saved weights, you can use the following code:
```
_, _, test_data = data_init()
batch_size = 20
model = torch.load('PATH_TO_MODEL/best_MODEL_NAME.pth', weights_only=False)
model.eval()
test_perp = perplexity(test_data, model, batch_size)
test_prep
```

