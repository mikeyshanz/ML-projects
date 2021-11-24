from tac_game import *
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from scipy.sparse import coo_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Activation
from keras.layers import Dropout

# Training each player with their input data
training_output = generate_training_rows(100)
x_1, y_1 = np.array(training_output[0]), np.array(training_output[1])
x_2, y_2 = np.array(training_output[2]), np.array(training_output[3])
del training_output  # reduce the load on memory
# Shuffling the training data
X_1_sparse, X_2_sparse = coo_matrix(x_1), coo_matrix(x_2)
x_1, X_1_sparse, y_1 = shuffle(x_1, X_1_sparse, y_1, random_state=0)
x_2, X_2_sparse, y_2 = shuffle(x_2, X_2_sparse, y_2, random_state=0)

# Here is your test and train sets son
x_train, y_train = x_1, y_1
x_test, y_test = x_2, y_2


def train_model(x_train, y_train, num_epochs_, batch_size_in_, output_size_, num_layers,
                node_sizes, dropout_perc_array, mid_activations, output_activation, loss_func,
                optimizer_in, layer_type_array, output_layer_type):
    if num_layers < 2:
        print("Need at least 2 layers")
        return None

    # Initializing the mode
    model = Sequential()

    # Creating the learning architecture
    model.add(layer_type_array[0](node_sizes[0], input_shape=(x_train.shape[1], )))
    model.add(Activation(mid_activations[0]))
    model.add(Dropout(dropout_perc_array[0]))

    for layer_ in range(1, num_layers-1):
        model.add(layer_type_array[layer_](node_sizes[layer_]))
        model.add(Activation(mid_activations[layer_]))
        model.add(Dropout(dropout_perc_array[layer_]))

    if num_layers == 2:
        layer_ = 1
    else:
        layer_ += 1

    model.add(layer_type_array[layer_](node_sizes[layer_]))
    model.add(Activation(mid_activations[layer_]))
    model.add(Dropout(dropout_perc_array[layer_]))

    model.add(output_layer_type(output_size_))
    model.add(Activation(output_activation))
    model.compile(loss=loss_func, optimizer=optimizer_in)

    temp_callback = model.fit(x_train, y_train,
                              epochs=num_epochs_, batch_size=batch_size_in_, verbose=0)

    return [model, temp_callback]


def get_single_accuracy(test_row, train_input, train_output, test_input, test_output):
    """
    This will take a set of hyperparameters and features and return the result of a
    very short training cycle. This is the fitness function of the genetic learning
    :param test_row: an array of length 10
    :return: returns the accuracy of the model, aka its fitness
    """
    trained_model = train_model(x_train=train_input, y_train=train_output, num_epochs_=10,
                                batch_size_in_=test_row[0], output_size_=len(train_output[0]),
                                num_layers=test_row[1], node_sizes=test_row[2], dropout_perc_array=test_row[3],
                                mid_activations=test_row[4], output_activation=test_row[5],
                                loss_func=test_row[6], optimizer_in=test_row[7],
                                layer_type_array=[eval(layer) for layer in test_row[9]],
                                output_layer_type=Dense)

    # temp_accuracy = trained_model[1].history['loss'][-1]
    try:
        temp_accuracy = r2_score(trained_model[0].predict(test_input), test_output)
    except ValueError:
        print("Model Predicted NaN values")
        return -10000
    del trained_model
    return temp_accuracy


possible_activations = ['relu', 'softmax', 'elu', 'selu', 'softplus', 'tanh', 'sigmoid', 'exponential', 'linear']
possible_optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax']
possible_loss_func = ['mean_squared_error', 'mean_absolute_error', 'squared_hinge', 'hinge', 'categorical_crossentropy',
                      'binary_crossentropy']
possible_layers = ['LSTM', 'GRU', 'Dense']
num_initial_layers = list(range(1, 4))
num_dense_layers = list(range(1, 10))

# Creating the hyper parameter input array randomly
# temp_batch_size = random.randrange(4, int(len(x_train) / 2))
# temp_num_layers = random.randrange(2, 10)
# temp_node_sizes = [2 ** random.randrange(2, 10) for _ in range(temp_num_layers)]
# temp_dropout = [random.randrange(0, 8) / 10 for _ in range(temp_num_layers)]
# temp_mid_activations = [possible_activations[random.randrange(len(possible_activations))] for _ in
#                         range(temp_num_layers)]
# temp_output_act = possible_activations[random.randrange(len(possible_activations))]
# temp_loss_func = possible_loss_func[random.randrange(len(possible_loss_func))]
# temp_optimizer = possible_optimizers[random.randrange(len(possible_optimizers))]
# temp_start_layer = possible_layers[random.randrange(2)]
# temp_layer_type_array = [temp_start_layer] + ['Dense' for _ in range(temp_num_layers - 1)]
#
# test_row = [temp_batch_size, temp_num_layers, temp_node_sizes, temp_dropout, temp_mid_activations,
#             temp_output_act, temp_loss_func, temp_optimizer, temp_start_layer, temp_layer_type_array]

accuracy_list, hyperset_list = list(), list()
best_accuracy = -1000000
for temp_num_layers in range(2, 8):
    for temp_output_act in possible_activations:
        for temp_loss_func in possible_loss_func:
            for temp_optimizer in possible_optimizers:
                # for temp_start_layer in possible_layers:
                print("Iteration...")
                # Set a specific test row to try
                temp_batch_size = 64
                # temp_num_layers = 2
                temp_node_sizes = [32 for _ in range(temp_num_layers)]
                temp_dropout = [.2 for _ in range(temp_num_layers)]
                temp_mid_activations = ['relu' for _ in range(temp_num_layers)]
                # temp_output_act = 'softmax'
                # temp_loss_func = 'mae'
                # temp_optimizer = 'adam'
                temp_start_layer = 'Dense'
                temp_layer_type_array = [temp_start_layer] + ['Dense' for _ in range(temp_num_layers - 1)]

                test_row = [temp_batch_size, temp_num_layers, temp_node_sizes, temp_dropout, temp_mid_activations,
                            temp_output_act, temp_loss_func, temp_optimizer, temp_start_layer, temp_layer_type_array]

                set_accuracy = get_single_accuracy(test_row=test_row, train_input=x_train,
                                                   train_output=y_train, test_input=x_test,
                                                   test_output=y_test)
                if set_accuracy > best_accuracy:
                    print("New Best Model! Accuracy:", set_accuracy)
                    best_accuracy = set_accuracy
                accuracy_list.append(set_accuracy)
                hyperset_list.append(test_row)

