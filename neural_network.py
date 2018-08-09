import numpy as np
import matplotlib.pyplot as plt

from activations import relu, sigmoid, backprop_sigmoid_unit, backprop_relu_unit, softmax

class NeuralNetworkImpl:
    def __init__(self, layer_sizes,
        layer_activations,
        optimization_algorithm='sgd',
        alpha=0.01,
        epochs=10000,
        mini_batch_size=64,
        regularization=0.0001,
        momentum_beta=0.9,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        rmsprop_beta1=0.9,
        rmsprop_epsilon=1e-8,
        plot_loss=True):

        self.layers_count = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.layer_activations = layer_activations
        self.optimization_algorithm = optimization_algorithm
        self.alpha = alpha
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.regularization = regularization
        self.momentum_beta = momentum_beta
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.rmsprop_beta1 = rmsprop_beta1
        self.rmsprop_epsilon = rmsprop_epsilon
        self.plot_loss = plot_loss

        self.last_layer_activation = self.layer_activations[self.layers_count - 1]

    def train(self, train_X, train_Y):
        n, m = train_X.shape
        num_params = self.__initialize_params(n)
        print("Parameters to train: {}".format(num_params))
        self.__initialize_optimizer()

        costs = []

        for i in range(self.epochs):
            permutation = list(np.random.permutation(m))
            shuffled_X = train_X[:, permutation]
            shuffled_Y = train_Y[:, permutation].reshape((train_Y.shape[0], m))

            epoch_costs = []
            epoch_accuracies = []
            accuracy = 0.0
            for j in range(0, m, self.mini_batch_size):
                batch_X = shuffled_X[:, j:j+self.mini_batch_size]
                batch_Y = shuffled_Y[:, j:j+self.mini_batch_size]
                y_hat = self.__forward_propagation(batch_X)
                cost = self.__compute_cost(y_hat, batch_Y)
                epoch_costs.append(cost)
                self.__backward_propagation(y_hat, batch_X, batch_Y)

                if i % 10 == 0 and j == 0:
                    if self.last_layer_activation == 'sigmoid':
                        accuracy = self.__logistic_accuracy(y_hat, batch_Y)
                    elif self.last_layer_activation == 'softmax':
                        accuracy = self.__multilabel_accuracy(y_hat, batch_Y)

            if i % 10 == 0:
                epoch_cost = np.average(epoch_costs)
                print("=== iteration {}, cost: {}, accuracy: {}".format(i, epoch_cost, accuracy))
                costs.append(epoch_cost)

        # plot the cost
        if self.plot_loss:
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('epochs')
            plt.title("Learning rate = " + str(self.alpha))
            plt.show()

    def __logistic_accuracy(self, y_hat, y):
        predictions = np.zeros(y.shape)

        for k in range(y.shape[1]):
            if y_hat[0,k] > 0.5:
                predictions[0,k] = 1
            else:
                predictions[0,k] = 0

        return np.sum((predictions == y) / y.shape[1])

    def __multilabel_accuracy(self, y_hat, y):
        predictions = np.zeros(y.shape)
        argmax_y_hat = np.argmax(y_hat, axis=0)
        argmax_y = np.argmax(y, axis=0)

        return np.sum((argmax_y_hat == argmax_y) / y.shape[1])

    def __initialize_params(self, features_count):
        number_params_to_train = 0
        np.random.seed(12)
        self.w = [ self.__layer_weights_initialization(features_count, self.layer_sizes[0]) ]
        self.b = [ np.zeros((self.layer_sizes[0], 1)) ]
        # Sum for W and b for the first layer
        number_params_to_train += (features_count + 1) * self.layer_sizes[0]

        for i in range(1, self.layers_count):
            self.w.append(self.__layer_weights_initialization(self.layer_sizes[i-1], self.layer_sizes[i]))
            self.b.append(np.zeros((self.layer_sizes[i], 1)))
            # Sum for W and b for this layer
            number_params_to_train += (self.layer_sizes[i-1] + 1) * self.layer_sizes[i]

        return number_params_to_train

    def __initialize_optimizer(self):
        if self.optimization_algorithm == 'sgd':
            pass # Do nothing
        elif self.optimization_algorithm == 'momentum':
            self.__initialize_momentum_optimizer()
        elif self.optimization_algorithm == 'adam':
            self.__initialize_adam_optimizer()
        elif self.optimization_algorithm == 'rmsprop':
            self.__initialize_rmsprop_optimizer()
        else:
            raise "Invalid optimization algorithm: {}".format(self.optimization_algorithm)

    def __initialize_momentum_optimizer(self):
        self.momentum_velocity_w = []
        self.momentum_velocity_b = []

        for l in range(self.layers_count):
            self.momentum_velocity_w.append(np.zeros(self.w[l].shape))
            self.momentum_velocity_b.append(np.zeros(self.b[l].shape))

    def __initialize_adam_optimizer(self):
        self.adam_counter = 0
        self.adamv_w = []
        self.adamv_b = []
        self.adams_w = []
        self.adams_b = []

        for l in range(self.layers_count):
            self.adamv_w.append(np.zeros(self.w[l].shape))
            self.adamv_b.append(np.zeros(self.b[l].shape))
            self.adams_w.append(np.zeros(self.w[l].shape))
            self.adams_b.append(np.zeros(self.b[l].shape))

    def __initialize_rmsprop_optimizer(self):
        self.rmsprop_w = []
        self.rmsprop_b = []

        for l in range(self.layers_count):
            self.rmsprop_w.append(np.zeros(self.w[l].shape))
            self.rmsprop_b.append(np.zeros(self.b[l].shape))

    def __layer_weights_initialization(self, prev_layer, curr_layer):
        return np.random.randn(curr_layer, prev_layer) * np.sqrt(2 / prev_layer)

    def __forward_propagation(self, x):
        self.cached_a = []
        self.cached_z = []

        a_prev = x
        for i in range(self.layers_count):
            a, z = self.__one_layer_forward_prop(a_prev, i, self.layer_activations[i])
            self.cached_a.append(a)
            self.cached_z.append(z)
            a_prev = a

        return a_prev

    def __one_layer_forward_prop(self, a_prev, layer_index, activation):
        z = np.dot(self.w[layer_index], a_prev) + self.b[layer_index]
        if activation == 'relu':
            return relu(z), z
        elif activation == 'sigmoid':
            return sigmoid(z), z
        elif activation == 'softmax':
            return softmax(z), z
        else:
            raise "Invalid activation: {}".format(activation)

    def __compute_cost(self, y_hat, y):
        _, m = y.shape
        regularization_cost = 0
        for i in range(self.layers_count):
            regularization_cost += np.linalg.norm(self.w[i])

        cost = 0.0
        if self.last_layer_activation == 'sigmoid':
            # When last layer's activation is sigmoid assume binary classification
            cost = (-1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) + (self.regularization / (2 * m)) * regularization_cost
        elif self.last_layer_activation == 'softmax':
            # When last layer's activation is softmax assume multi-class classification
            cost = (-1 / m) * np.sum(y * np.log(y_hat)) + (self.regularization / (2 * m)) * regularization_cost
        else:
            raise "Invalid last layer activation: {}".format(self.last_layer_activation)

        return np.squeeze(cost)

    def __one_layer_backward_prop(self, y_hat, x, y, prev_dz, layer_index):
        _, m = y.shape

        if layer_index == 0:
            prev_a = x
        else:
            prev_a = self.cached_a[layer_index - 1]

        if layer_index == self.layers_count - 1:
            # In this implementation of a neural network we will always use some
            # form of cross-entropy loss (logistic loss) and in both cases when we
            # have either binary classification or multi-label classification the
            # derivative of this loss function with respect to the values computed
            # before the activation is y_hat - y.
            # https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
            if self.last_layer_activation not in ['sigmoid', 'softmax']:
                raise "Invalid last layer activation: {}".format(self.last_layer_activation)
            dz = y_hat - y
        else:
            da = np.dot(self.w[layer_index+1].T, prev_dz)
            if self.layer_activations[layer_index] == 'relu':
                dz = da * backprop_relu_unit(self.cached_z[layer_index])
            elif self.layer_activations[layer_index] == 'sigmoid':
                dz = da * backprop_sigmoid_unit(self.cached_z[layer_index])
            else:
                raise "Not supported layer activation: {}".format(self.layer_activations[layer_index])

        dw = (1 / m) * np.dot(dz, prev_a.T) + (self.regularization / m) * self.w[layer_index]
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        return dz, dw, db

    def __backward_propagation(self, y_hat, x, y):
        dz = [None] * self.layers_count
        dw = [None] * self.layers_count
        db = [None] * self.layers_count

        for i in reversed(range(self.layers_count)):
            prev_dz = None
            if i < self.layers_count - 1:
                prev_dz = dz[i + 1]
            dz[i], dw[i], db[i] = self.__one_layer_backward_prop(y_hat, x, y, prev_dz, i)

        # Update the params
        self.__update_parameters(dw, db)

    def __update_parameters(self, dw, db):
        if self.optimization_algorithm == 'sgd':
            self.__update_parameters_with_sgd(dw, db)
        elif self.optimization_algorithm == 'momentum':
            self.__update_parameters_with_momentum(dw, db)
        elif self.optimization_algorithm == 'adam':
            self.__update_parameters_with_adam(dw, db)
        elif self.optimization_algorithm == 'rmsprop':
            self.__update_parameters_with_rmsprop(dw, db)
        else:
            raise "Invalid optimization algorithm: {}".format(self.optimization_algorithm)

    def __update_parameters_with_sgd(self, dw, db):
        for i in range(self.layers_count):
            self.w[i] = self.w[i] - self.alpha * dw[i]
            self.b[i] = self.b[i] - self.alpha * db[i]

    def __update_parameters_with_momentum(self, dw, db):
        for i in range(self.layers_count):
            self.momentum_velocity_w[i] = self.momentum_beta * self.momentum_velocity_w[i] + (1 - self.momentum_beta) * dw[i]
            self.momentum_velocity_b[i] = self.momentum_beta * self.momentum_velocity_b[i] + (1 - self.momentum_beta) * db[i]

            self.w[i] = self.w[i] - self.alpha * self.momentum_velocity_w[i]
            self.b[i] = self.b[i] - self.alpha * self.momentum_velocity_b[i]

    def __update_parameters_with_adam(self, dw, db):
        self.adam_counter += 1
        for i in range(self.layers_count):
            self.adamv_w[i] = self.adam_beta1 * self.adamv_w[i] + (1 - self.adam_beta1) * dw[i]
            adamv_corrw = self.adamv_w[i] / (1 - self.adam_beta1 ** self.adam_counter)
            self.adamv_b[i] = self.adam_beta1 * self.adamv_b[i] + (1 - self.adam_beta1) * db[i]
            adamv_corrb = self.adamv_b[i] / (1 - self.adam_beta1 ** self.adam_counter)

            self.adams_w[i] = self.adam_beta2 * self.adams_w[i] + (1 - self.adam_beta2) * (dw[i] * dw[i])
            adams_corrw = self.adams_w[i] / (1 - self.adam_beta2 ** self.adam_counter)
            self.adams_b[i] = self.adam_beta2 * self.adams_b[i] + (1 - self.adam_beta2) * (db[i] * db[i])
            adams_corrb = self.adams_b[i] / (1 - self.adam_beta2 ** self.adam_counter)

            self.w[i] = self.w[i] - self.alpha * (adamv_corrw / (np.sqrt(adams_corrw) + self.adam_epsilon))
            self.b[i] = self.b[i] - self.alpha * (adamv_corrb / (np.sqrt(adams_corrb) + self.adam_epsilon))

    def __update_parameters_with_rmsprop(self, dw, db):
        for i in range(self.layers_count):
            self.rmsprop_w[i] = self.rmsprop_beta1 * self.rmsprop_w[i] + (1 - self.rmsprop_beta1) * (dw[i] * dw[i])
            self.rmsprop_b[i] = self.rmsprop_beta1 * self.rmsprop_b[i] + (1 - self.rmsprop_beta1) * (db[i] * db[i])

            self.w[i] = self.w[i] - self.alpha * (dw[i] / (np.sqrt(self.rmsprop_w[i]) + self.rmsprop_epsilon))
            self.b[i] = self.b[i] - self.alpha * (db[i] / (np.sqrt(self.rmsprop_b[i]) + self.rmsprop_epsilon))

    def predict(self, X):
        a = self.__forward_propagation(X)
        if self.last_layer_activation == 'sigmoid':
            return a > 0.5
        elif self.last_layer_activation == 'softmax':
            predictions = np.zeros(a.shape)
            max_indexes = np.argmax(a, axis=0)
            for i in range(predictions.shape[1]):
                predictions[max_indexes[i],i] = 1
            return predictions
        else:
            raise "Invalid last layer activation: {}".format(self.last_layer_activation)

    def predict_raw(self, X):
        return self.__forward_propagation(X)
