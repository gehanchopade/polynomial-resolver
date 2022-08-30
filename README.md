## Create a new environment in conda

Follow the following steps

## Make sure the environment has cudatoolkit and cudnn installed.

1. conda create --name tensorflow-gpu python=3.8
2. conda activate tensorflow-gpu
3. conda install -c anaconda tensorflow-gpu cudatoolkit
4. conda env update --file tools.yml
5. pip install -r requirements.txt

test.txt is a file of lines chosen randomly from train.txt and used solely for testing.

training script present in main.py

Network details present in network.txt - Network is a simple LSTM based model.

Model is trained on 30 epochs and is saved in factors_expansion.h5

Model gets an accuracy of 82.85% on the given test case. Accuracy can be improved by testing for more number of epochs. There is more scope for convergence.
