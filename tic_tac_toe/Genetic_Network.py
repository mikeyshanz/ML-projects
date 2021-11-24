from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Activation
from keras.layers import Dropout
import random
import psutil
from tac_game import *
from sklearn.utils import shuffle
# from sklearn.metrics import r2_score
from scipy.sparse import coo_matrix


def train_model(x_train, y_train, num_epochs_, batch_size_in_, input_size_, output_size_, num_layers,
                node_sizes, dropout_perc_array, mid_activations, output_activation, loss_func,
                optimizer_in, layer_type_array, output_layer_type):

    # Initializing the mode
    model = Sequential()

    # Creating the learning architecture
    model.add(layer_type_array[0](node_sizes[0], input_shape=(input_size_, 1)))
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
                              epochs=num_epochs_, batch_size=batch_size_in_, verbose=1)

    return [model, temp_callback]


def get_single_accuracy(test_row):
    """
    This will take a set of hyperparameters and features and return the result of a
    very short training cycle. This is the fitness function of the genetic learning
    :param test_row: an array of length 10
    :return: returns the accuracy of the model, aka its fitness
    """
    trained_model = train_model(x_train=train_input, y_train=train_output, num_epochs_=10,
                                batch_size_in_=test_row[0],
                                input_size_=len(train_input[0]), output_size_=len(train_output[0]),
                                num_layers=test_row[1], node_sizes=test_row[2], dropout_perc_array=test_row[3],
                                mid_activations=test_row[4], output_activation=test_row[5],
                                loss_func=test_row[6], optimizer_in=test_row[7],
                                layer_type_array=[eval(layer) for layer in test_row[9]],
                                output_layer_type=Dense)

    temp_accuracy = trained_model[1].history['loss'][-1]
    del trained_model
    return temp_accuracy


def get_population_accuracy(all_inputs):
    accuracy_list = []
    worked_inputs = []
    for idx, input_ in enumerate(all_inputs):
        print("Solving for child #", idx+1)
        print("Current RAM is :", psutil.virtual_memory()[2], "%\n")
        try:
            temp_acc = get_single_accuracy(input_)
            accuracy_list.append(temp_acc)
            worked_inputs.append(input_)
            print("Best Generation Accuracy:", min(accuracy_list))
        except:
            print("Stuff didn't work fam :(")

    return [accuracy_list, worked_inputs]


def create_initial_population(population_size):
    all_test_rows, all_accuracies = [], []

    while len(all_test_rows) < population_size:
        temp_batch_size = random.randrange(4, int(len(train_input)/2))
        temp_num_layers = random.randrange(2, 10)
        temp_node_sizes = [2**random.randrange(2, 10) for _ in range(temp_num_layers)]
        temp_dropout = [random.randrange(0, 8)/10 for _ in range(temp_num_layers)]
        temp_mid_activations = [possible_activations[random.randrange(len(possible_activations))] for _ in range(temp_num_layers)]
        temp_output_act = possible_activations[random.randrange(len(possible_activations))]
        temp_loss_func = possible_loss_func[random.randrange(len(possible_loss_func))]
        temp_optimizer = possible_optimizers[random.randrange(len(possible_optimizers))]
        temp_start_layer = possible_layers[random.randrange(2)]
        temp_layer_type_array = [temp_start_layer] + ['Dense' for _ in range(temp_num_layers-1)]

        test_row = [temp_batch_size, temp_num_layers, temp_node_sizes, temp_dropout, temp_mid_activations,
                    temp_output_act, temp_loss_func, temp_optimizer, temp_start_layer, temp_layer_type_array]

        temp_accuracy = get_single_accuracy(test_row)
        if not np.isnan(temp_accuracy):
            print("Population Size:", len(all_test_rows))
            all_test_rows.append(test_row)
            all_accuracies.append(temp_accuracy)

    return [all_test_rows, all_accuracies]


def mutate_feature(feature_idx, temp_num_layers):
    temp_batch_size = random.randrange(4, int(len(train_input) / 2))
    # temp_num_layers = random.randrange(2, 10)
    temp_node_sizes = [2 ** random.randrange(2, 10) for _ in range(temp_num_layers)]
    temp_dropout = [random.randrange(0, 8) / 10 for _ in range(temp_num_layers)]
    temp_mid_activations = [possible_activations[random.randrange(len(possible_activations))] for _ in
                            range(temp_num_layers)]
    temp_output_act = possible_activations[random.randrange(len(possible_activations))]
    temp_loss_func = possible_loss_func[random.randrange(len(possible_loss_func))]
    temp_optimizer = possible_optimizers[random.randrange(len(possible_optimizers))]
    temp_start_layer = possible_layers[random.randrange(2)]
    temp_layer_type_array = [temp_start_layer] + ['Dense' for _ in range(temp_num_layers - 1)]

    test_row = [temp_batch_size, temp_num_layers, temp_node_sizes, temp_dropout, temp_mid_activations,
                temp_output_act, temp_loss_func, temp_optimizer, temp_start_layer, temp_layer_type_array]

    return test_row[feature_idx]


def create_children(prev_all_test_rows, prev_all_accuracies, population_size):
    # Take the top 50% of the population and create childen based on random peices of their input rows
    all_children, best_inputs = [], []

    half_idx = int(len(prev_all_accuracies) / 2)
    for row in range(half_idx):
        best_inputs.append(prev_all_test_rows[np.argsort(prev_all_accuracies)[row]])

    while len(all_children) < population_size:
        rand_input1, rand_input2 = 0, 0
        while rand_input1 == rand_input2:
            rand_input1 = best_inputs[random.randrange(len(best_inputs))]
            rand_input2 = best_inputs[random.randrange(len(best_inputs))]

        child = []
        for input_portion in range(len(rand_input1)):
            rand_num = random.randrange(2)
            if rand_num == 1:
                child.append(rand_input1[input_portion])
            else:
                child.append(rand_input2[input_portion])

        # If the number of layers is different, this will make sure all inputs dependent on the number of layers
        # is lined up appropritaly.

        if rand_input1[1] != rand_input2[1]:
            child[2] = mutate_feature(feature_idx=2, temp_num_layers=child[1])
            child[3] = mutate_feature(feature_idx=3, temp_num_layers=child[1])
            child[4] = mutate_feature(feature_idx=4, temp_num_layers=child[1])
            child[9] = mutate_feature(feature_idx=9, temp_num_layers=child[1])

        all_children.append(child)

    return all_children


def mutate_population(new_input_rows, row_mutation_likliehood, feature_mutation_likliehoood):
    mutated_population = []
    for row in range(len(new_input_rows)):
        row_chance = random.random()
        if row_chance <= row_mutation_likliehood:
            temp_mutated_row = []
            for feature in range(len(new_input_rows[row])):
                feature_chance = random.random()
                if feature_chance <= feature_mutation_likliehoood and feature != 1:  # Do not change the num_layers
                    temp_mutated_row.append(mutate_feature(feature, new_input_rows[row][1]))
                else:
                    temp_mutated_row.append(new_input_rows[row][feature])
            mutated_population.append(temp_mutated_row)
        else:
            mutated_population.append(new_input_rows[row])

    return mutated_population


def learn_genetically(population_size_in, num_generations, row_mutation_like_in, feature_mutation_like_in):
    print("Getting initial population...")
    initial_input_pop, initial_accuracies = create_initial_population(population_size=population_size_in)

    first_children = create_children(prev_all_test_rows=initial_input_pop, prev_all_accuracies=initial_accuracies,
                                     population_size=population_size_in)

    mutated_generation = mutate_population(new_input_rows=first_children, row_mutation_likliehood=row_mutation_like_in,
                                           feature_mutation_likliehoood=feature_mutation_like_in)

    mutated_generation_accuracy, mutated_generation_inputs = get_population_accuracy(mutated_generation)

    non_nan_inputs, non_nan_accuracies = [], []
    for idx, temp_acc in enumerate(mutated_generation_accuracy):
        if not np.isnan(temp_acc):
            non_nan_inputs.append(mutated_generation_inputs[idx])
            non_nan_accuracies.append(temp_acc)

    best_error_rates = list()
    best_inputs = list()
    best_error_rates.append(np.min(non_nan_accuracies))
    best_inputs.append(non_nan_inputs[non_nan_accuracies.index(np.min(non_nan_accuracies))])
    print("Starting Error Rate:", best_error_rates[0])

    del first_children
    del mutated_generation
    del mutated_generation_accuracy
    del mutated_generation_inputs

    for generation in range(num_generations):
        print("Generation:", generation+1)
        next_children = create_children(prev_all_test_rows=non_nan_inputs, prev_all_accuracies=non_nan_accuracies,
                                        population_size=population_size_in)

        next_mutated_generation = mutate_population(new_input_rows=next_children, row_mutation_likliehood=row_mutation_like_in,
                                                    feature_mutation_likliehoood=feature_mutation_like_in)

        next_mutated_generation_acc, next_mutated_generation_inputs = get_population_accuracy(next_mutated_generation)

        non_nan_inputs, non_nan_accuracies = [], []
        for idx, temp_acc in enumerate(next_mutated_generation_acc):
            if not np.isnan(temp_acc):
                non_nan_inputs.append(next_mutated_generation_inputs[idx])
                non_nan_accuracies.append(temp_acc)

        print("Best Error Rate:", np.min(best_error_rates))
        print("Current RAM Usage:", psutil.virtual_memory()[2], "%\n")
        best_error_rates.append(np.min(non_nan_accuracies))
        best_inputs.append(non_nan_inputs[non_nan_accuracies.index(np.min(non_nan_accuracies))])

        del next_children
        del next_mutated_generation
        del next_mutated_generation_acc
        del next_mutated_generation_inputs

    return [best_error_rates, best_inputs]


# feedback = get_feedback(only_unique_feedback=False)
# x_list, y_list = create_all_training(feedback=feedback, row_limit=200, feedback_stop_list=['Automatically Archived'])
# del feedback

# Training each player with their input data
training_output = generate_training_rows(100)
x_1, y_1 = np.array(training_output[0]), np.array(training_output[1])
x_2, y_2 = np.array(training_output[2]), np.array(training_output[3])
del training_output  # reduce the load on memory
# Shuffling the training data
X_1_sparse, X_2_sparse = coo_matrix(x_1), coo_matrix(x_2)
x_1, X_1_sparse, y_1 = shuffle(x_1, X_1_sparse, y_1, random_state=0)
x_2, X_2_sparse, y_2 = shuffle(x_2, X_2_sparse, y_2, random_state=0)
training_perc = 0.9

train_limit = int(len(x_1)*training_perc)
train_input = np.array([np.array(input_row_).reshape(-1, 1) for input_row_ in x_1][0:train_limit])
train_output = np.array([np.array(output_row_).reshape(-1, ) for output_row_ in y_1][0:train_limit])

test_input = [np.array(input_row_).reshape(-1, 1) for input_row_ in x_1][train_limit:len(x_1)]
test_output = [np.array(output_row_).reshape(-1, ) for output_row_ in y_1][train_limit:len(x_1)]


possible_activations = ['relu', 'softmax', 'elu', 'selu', 'softplus', 'tanh', 'sigmoid', 'exponential', 'linear']
possible_optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax']
possible_loss_func = ['mean_squared_error', 'mean_absolute_error', 'squared_hinge', 'hinge', 'categorical_crossentropy',
                      'binary_crossentropy']
possible_layers = ['LSTM', 'GRU', 'Dense']

print("Starting the Learning...")
best_errors, best_input_arrays = learn_genetically(population_size_in=4, num_generations=2, row_mutation_like_in=0.3,
                                                   feature_mutation_like_in=0.2)

