#!/usr/bin/env python3

import random
import csv
import matplotlib.pyplot as plt
from math import exp, floor, ceil

def stochastic_gradient_descent(network, classes, training_data):
    for _ in range(0, EPOCHS):
        first_example = True
        total_error = 0.00
        for example in training_data:
            temporal_delta = [neuron['d'] \
                for layer in network for neuron in layer] \
                if not first_example else None
            outputs = [0 for _ in range(classes)]
            outputs[int(example[-1])] = 1
            actual = feed_forward(network, example)
            total_error += sse(actual, outputs)
            backpropagate(network, outputs)
            update_weights(network, example, temporal_delta)
            reset_neurons(network)
            first_example = False
        MSE.append(total_error/len(training_data))
        TRP.append(performance_measure(NETWORK, TRAIN))
        TEP.append(performance_measure(NETWORK, TEST))

def feed_forward(network, example):
    layer_input, layer_output = example, []
    for layer in network:
        for neuron in layer:
            summ = summing_function(neuron['w'], layer_input)
            neuron['o'] = activation_function(summ)
            layer_output.append(neuron['o'])
        layer_input, layer_output = layer_output, []
    return layer_input

def backpropagate(network, example):
    for i in range(len(network)-1, -1, -1):
        for j in range(len(network[i])):
            err = 0.00
            if i == len(network)-1:
                err = example[j] - network[i][j]['o']
            else:
                summ = 0.00
                for neuron in network[i+1]:
                    summ += neuron['w'][j] * neuron['d']
                err = summ
            network[i][j]['d'] = activation_derivative(network[i][j]['o']) * err

def reset_neurons(network):
    for layer in network:
        for neuron in layer:
            neuron['o'] = 0

def update_weights(network, example, delta):
    for i in range(len(network)):
        if i != 0:
            t = [neuron['o'] for neuron in network[i-1]]
        else:
            t = example[:-1]
        for neuron, d in zip(network[i], range(0, len(network[i]))):
            for f in range(len(t)):
                neuron['w'][f] += LEARNING_RATE * float(t[f]) * neuron['d']
                if delta is not None:
                    neuron['w'][f] += MOMENTUM_RATE * delta[d]
                neuron['w'][-1] += LEARNING_RATE * neuron['d']

def sse(actual, target):
    summ = 0.00
    for i in range(len(actual)):
        summ += (actual[i] - target[i])**2
    return summ

def activation_function(z):
    numerator = 1
    denominator = 1 + exp(-z)
    return numerator/denominator

def activation_derivative(z):
    return z * (1 - z)

def summing_function(weights, inputs):
    bias = weights[-1]
    summ = 0.00
    for i in range(len(weights)-1):
        summ += (weights[i] * float(inputs[i]))
    return summ + bias

def performance_measure(network, data):
    correct, total = 0, 0
    for example in data:
        if check_output(network, example) == float(example[-1]):
            correct += 1
        total += 1
    return 100*(correct / total)

def check_output(network, example):
    output = feed_forward(network, example)
    return output.index(max(output))

def initialize_network(n, h, o):
    def r():
        return random.uniform(-0.50, 0.50)
    neural_network = []
    neural_network.append([{'w':[r() for i in range(n+1)]} for j in range(h)])
    neural_network.append([{'w':[r() for i in range(h+1)]} for j in range(o)])
    return neural_network

def load_data(filename):
    with open(filename, newline='\n') as csv_file:
        data = []
        rows = csv.reader(csv_file, delimiter=',')
        for row in rows:
            data.append(row)
    random.shuffle(data)
    training_data = data[:floor(len(data)*0.70)]
    testing_data = data[-ceil(len(data)*0.30):]
    return training_data, testing_data

def plot_data():
    x = range(0, EPOCHS)
    fig, ax2 = plt.subplots()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE', color='blue')
    line, = ax2.plot(x, MSE, '-', c='blue', lw='1', label='MSE')
    ax1 = ax2.twinx()
    ax1.set_ylabel('Accuracy', color='green')
    line2, = ax1.plot(x, TRP, '-', c='green', lw='1', label='Training')
    line3, = ax1.plot(x, TEP, ':', c='green', lw='1', label='Testing')
    fig.tight_layout()
    fig.legend(loc='center')
    plt.show()
    plt.clf()

if __name__ == '__main__':
    TRAIN, TEST = load_data('../data/iris.csv')
    FEATURES = len(TRAIN[0][:-1])
    CLASSES = len(list(set([c[-1] for c in TRAIN])))
    NETWORK = initialize_network(FEATURES, 5, CLASSES)
    LEARNING_RATE, MOMENTUM_RATE = 0.100, 0.001
    EPOCHS = 200
    MSE, TRP, TEP = [], [], []
    stochastic_gradient_descent(NETWORK, CLASSES, TRAIN)
    plot_data()
