# -*- coding:utf-8 -*-
# @Filename:    Tensorflow_flow.py
# Created on:   09/10/21 10:33
# @Author:      Luc


import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns


def exercise_1():
    """
        1. Make sure tensorflow is installed on your environment: 'conda install
        tensorflow'
        2. import tensorflow as tf 3. Check your version of tf, use print(tf.__version__)
        4. Create a constant node (node1) and a Tensor node (node2) with the values [1,2,3,4,5]
        and [1,1,2,3,5]
        5. Perform an element-wise multiplication of the two nodes and store it to node3,
        Print the value of node3, use the .numpy() method
        6. Sum the values of the elements in node3, store the result in node4.
    :return:
    """
    print(tf.__version__)
    node1 = tf.constant([1, 2, 3, 4, 5])
    node2 = tf.constant([1, 1, 2, 3, 5])
    print(f'node 1: {node1}, node 2: {node2}')
    node3 = node1 + node2
    print(f'node3 = node1 + node2: {node3}')
    node4 = tf.math.reduce_sum(node3)
    print(f'node4 = sum elements of node3: {node4}')


def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


def get_dataset_batches(x, y, batch_size):
    n_samples = x.shape[0]
    num_of_batches = int(n_samples / batch_size)
    enough_samples = (num_of_batches > 2)
    data_set = tf.data.Dataset.from_tensor_slices((x, y))
    data_set = data_set.shuffle(buffer_size=sys.getsizeof(data_set))
    if enough_samples:
        data_set = data_set.batch(batch_size)

    return data_set


class NN(tf.Module):
    """
    Fully connected Neural Network with TensorFLow
    """
    def __init__(self, input_size, layers, loss_f=loss, name=None):
        """
        :param input_size: number of features the NN class is getting as input
        :param layers: list of tuples (number of neuron, activation function of layer) in each layer.
         For example: layers = [(2, tf.nn.leaky_relu), (4, tf.nn.relu)] means that the network
         has 2 layers the first one with 2 neuron and activation of leaky_relu the the last layer (num 2) is
         with 4 neurons (its weights matrix will be of dimension of 2x4) and relu activation function
        :param loss_f: loss function address
        :param name: custom name of the network
        """
        super(NN, self).__init__(name=name)
        self.layers = []
        self.loss = loss
        with self.name_scope:
            for n_neurons, f_a in layers:
                self.layers.append(Layer(input_dim=input_size, output_dim=n_neurons, f_activation=f_a))
                input_size = n_neurons

    # @tf.Module.with_name_scope
    def __call__(self, x):
        # forward pass
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, item):
        return self.layers[item]

    def fit(self, x, y, epochs=20, batch_size=32, l_r=0.01):

        # slice x,y into batches
        dataset = get_dataset_batches(x, y, batch_size)
        loss_values = []

        for epoch in range(epochs):

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(dataset):

                # Open a GradientTape to record the operations run
                with tf.GradientTape() as tape:
                    # Run the forward pass
                    # The operations are going to be recorded
                    # on the GradientTape thanks to tf.Module.variables.
                    y_hats = self.__call__(x_batch_train)

                    # Compute the loss value for this batch.
                    loss_value = self.loss(y_batch_train, y_hats)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self.trainable_variables)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                for d_v, v in zip(grads, self.trainable_variables):
                    v.assign_sub(l_r * d_v)

                # Log every  batches.
                if step % 8 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print(f"Seen so far: {((step + 1) * batch_size)} samples")
            loss_values.append(loss_value)

            print("Epoch: %2d loss=%2.5f" % (epoch, loss_value))
        return loss_values

    def predict(self, x):
        return self.__call__(x)

    def __str__(self):
        return f"{self.name}, num of layers: {len(self.layers)}"


class Layer(tf.Module):
    def __init__(self, input_dim, output_dim, f_activation=tf.nn.leaky_relu, name=None):
        """
        init the dimension of Layer class
        :param input_dim: represent n_features is first layer otherwise number of neuron in previous layer
        :param output_dim: number of neurons in layer
        :param f_activation: activation function
        :param name:
        """
        super(Layer, self).__init__(name=name)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.f_a = f_activation
        # bias
        self.b = None
        # previous activation @ self.w
        self.z = None
        # activation(self.z)
        self.a = None
        # weights (self.input_dim x self.out_dim)
        self.w = None
        self._build(input_dim, output_dim)

    def _build(self, input_dim, output_dim):
        """
        initialize the layer's weights according to input_shape.
        For example: if input shape is (2,3) it would init self.weights with (3, self.units) tensor  random values
        from normal distribution mean=0 std=0.05
        :param input_shape: input shape of the previous layer
        :return: self.weights, self.b
        """
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
        b_init = tf.zeros_initializer()

        self.w = tf.Variable(w_init([input_dim, output_dim]), name='weights')
        self.b = tf.Variable(b_init([output_dim]), name='bias')

    def __call__(self, x):
        self.z = tf.matmul(x, self.w) + self.b
        self.a = self.f_a(self.z)
        return self.a

    def __str__(self):
        rows, cols = self.w.shape
        return f"{self.name}, input: {rows}, out: {cols}"


def identity(x):
    return x


def exercise_2():
    """
        In this exercise you will define a simple Linear Regression model with TensorFlow low-level API.
        For a Linear Model y = Wx+b the graph looks like this:
        7. Load the data from the file “data_for_linear_regression_tf.csv".
        8. Define a class called MyModel()
        9. The class should have two Variables (W and b), and a call method that returns the model's
            output for a given input value. call method - def __call__(self, x)
        10. Define a loss method that receives the predicted_y and the target_y as
            arguments and returns the loss value for the current prediction. Use mean square error loss
        11. Define a train() method that does five train cycles of your model (i.e. 5 epochs)
            a. use tf.GradientTape() and the loss function that you have defined to record the loss as the linear
            operation is processed by the network.
            b. use the tape.gradient() method to retrieve the derivative of the loss with respect to W and b (dW, db)
            and update W and b accordingly
            12. Now, use the data to train and test your model:
                a. Train your model for 100 epochs, with learning_rate 0.1
                b. Save your model's W and b after each epoch, store results in a list for plotting purposes.
                Print the W, b and loss values after training
    :return:
    """
    data = pd.read_csv('data_for_linear_regression_tf.csv')
    x = tf.constant(data[['x']].values, dtype=tf.float32)
    y = tf.Variable(data[['y']].values, dtype=tf.float32)
    # no need activation as it is a linear problem
    reg_nn = NN(1, [(2, identity), (1, identity)], "Regression")
    loss_history = reg_nn.fit(x, y, epochs=200, batch_size=32, l_r=0.001)
    metrics = pd.DataFrame({"Loss": [loss.numpy() for loss in loss_history]})
    data['y_pred'] = reg_nn.predict(x).numpy()
    # plt.figure()
    # gca stands for 'get current axis'
    ax = plt.gca()

    data.plot(kind='scatter', x='x', y='y', ax=ax)
    data.plot(kind='scatter', x='x', y='y_pred', color='red', ax=ax)

    metrics.plot()
    # data.plot()
    plt.show()
    print(f"\n exercise_2 w: {reg_nn.trainable_variables}, loss: {reg_nn.loss(y, data['y_pred'].values)}")



def exercise_3():
    """
    In todays exercise we will build a neural network to predict avocado prices from the
    following dataset:
    Avocado Prices | Kaggle
    2. We can use TensorFlow only (no Keras today), and build a neural network of as
    many layers as we wish. Use a GradientTape to store the gradients.
    3. We can play with the learning rate, and play with any other parameter we want
    :return:
    """

    def process_data(avocado):
        ds = avocado.copy()

        # labels
        labels = ds[['AveragePrice']]
        # drop unnecessary features
        ds = ds.drop(columns=['Date', 'AveragePrice'])

        # process data - categorical
        cat_cols = ds.columns[(ds.dtypes == 'object')]
        ds[cat_cols] = ds[cat_cols].astype('category')

        # encode categories
        for feature in cat_cols:
            ds[feature] = ds[feature].cat.codes

        # encode year
        label_enc = LabelEncoder()
        ds['year'] = label_enc.fit_transform(ds['year'])

        # scale float features as change to float32 for tensor object
        num_cols = ds.columns[(ds.dtypes == np.number)]
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        ds[num_cols] = scaler.fit_transform(ds[num_cols])


        return ds, labels

    avocado = pd.read_csv('avocado.csv')
    data, labels = process_data(avocado)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.12, random_state=42)
    # crate tensor obj
    x_train = tf.constant(X_train.values, dtype=tf.float32)
    y_train = tf.Variable(y_train.values, dtype=tf.float32)
    x_test = tf.constant(X_test.values, dtype=tf.float32)

    avocado_model = NN(input_size=x_train.shape[-1],
                       layers=[(128, tf.nn.tanh), (128, tf.nn.leaky_relu), (64, tf.nn.relu), (1, identity)],
                       name='avocado')

    loss_history = avocado_model.fit(x_train, y_train, epochs=150, batch_size=128, l_r=0.0001)

    metrics = pd.DataFrame({"Loss": [loss.numpy() for loss in loss_history]})
    metrics.plot()
    plt.show()

    # run predictions on the test
    result = y_test.copy()
    result['AveragePrice_Predict'] = avocado_model.predict(x_test).numpy()
    sns.pairplot(result)
    plt.show()

    loss_pred = loss(result['AveragePrice'].values, result['AveragePrice_Predict'].values)
    print(f"\n model: {avocado_model} loss: {loss_pred}")




if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    exercise_1()
    exercise_2()
    exercise_3()
    plt.show(block=True)
