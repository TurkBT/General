import pandas as pd
import numpy as np


data = pd.read_pickle(r'C:\Users\batuh\PycharmProjects\project1\mnist.pkl')
# Preparing the datas
training_value = np.transpose(np.array(data[0][0]))
training_label = np.transpose((np.array(data[0][1]).reshape(-1, 1)))
# training_data = np.hstack((training_label, training_value))


test_value = np.transpose(np.array(data[2][0]))
test_label = np.transpose(np.array(data[2][1]).reshape(-1, 1))
# test_data = np.hstack((test_label, test_value))


class NeuralNetwork:
    def __init__(self):
        self.weight_1 = np.random.randn(neuron_number, 784)
        self.bias_1 = np.random.randn(neuron_number, 1)
        self.weight_2 = np.random.randn(10, neuron_number)
        self.bias_2 = np.random.randn(10, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, input_data):
        conv1 = self.weight_1.dot(input_data) + self.bias_1
        act1 = self.sigmoid(conv1)
        conv2 = self.weight_2.dot(act1) + self.bias_2
        act2 = self.sigmoid(conv2)
        return conv1, conv2, act1, act2

    def one_hot(self, label):
        one_hot_l = np.zeros((label.size, label.max() + 1))
        one_hot_l[np.arange(label.size), label] = 1
        return np.transpose(one_hot_l)

    def sigmoid_derivative(self, x):
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def backward_propagation(self, conv1, act1, act2, input_data, label_data):
        m = label_data.size
        one_hot_l = self.one_hot(label_data)
        dconv2 = act2 - one_hot_l
        dweight_2 = 1 / m * dconv2.dot(np.transpose(act1))
        dbias_2 = 1 / m * np.sum(dconv2)
        # dbias_2 = 1 / m * np.sum(dZ2, axis=1).reshape(-1, 1)
        dconv1 = np.transpose(self.weight_2).dot(dconv2) * self.sigmoid_derivative(conv1)
        dweight_1 = 1 / m * dconv1.dot(np.transpose(input_data))
        dbias_1 = 1 / m * np.sum(dconv1)
        return dweight_1, dweight_2, dbias_1, dbias_2

    def update_param(self, alpha, dweight_1, dweight_2, dbias_1, dbias_2):
        self.weight_1 = self.weight_1 - alpha * dweight_1
        self.weight_2 = self.weight_2 - alpha * dweight_2
        self.bias_1 = self.bias_1 - alpha * dbias_1
        self.bias_2 = self.bias_2 - alpha * dbias_2

    def get_predictions(self, act2):
        return np.argmax(act2, 0)

    def get_accuracy(self, predictions, label):
        print(predictions)
        return np.sum(predictions == label) / label.size

    def gradient_descent(self, data, label, epochs, alpha, mini_size):
        data_size = len(np.transpose(data))
        # weight1, weight2, bias1, bias2 = self.weight_1, self.weight_2, self.bias_1, self.bias_2
        for i in range(epochs):
            num_rows = np.transpose(data).shape[0]
            indices = np.random.permutation(num_rows)
            shuffled_data = np.transpose(data)[indices]
            shuffled_label = np.transpose(label)[indices]
            for j in range(data_size // mini_size):
                start = j * mini_size
                end = (j + 1) * mini_size
                sub_data = np.transpose(shuffled_data[start:end, :])
                sub_label = np.transpose(shuffled_label[start:end, :].reshape(-1, 1))
                conv1, conv2, act1, act2 = self.forward_propagation(sub_data)
                dweight1, dweight2, dbias1, dbias2 = self.backward_propagation(conv1, act1, act2, sub_data, sub_label)
                self.update_param(alpha, dweight1, dweight2, dbias1, dbias2)
                if j % 10 == 0:
                    print("Epochs: ", i)
                    print("Mini_Batch_Number: ", j)
                    print("Accuracy: ", self.get_accuracy(self.get_predictions(act2), sub_label))


neuron_number = 30

fnn = NeuralNetwork()

epochs = 1000
mini_bach_size = 10000

fnn.gradient_descent(training_value, training_label, epochs, 0.5, mini_bach_size)

final_conv1, final_conv2, final_act1, final_act2 = fnn.forward_propagation(test_value)
print("Accuracy: ", fnn.get_accuracy(fnn.get_predictions(final_act2), test_label))
