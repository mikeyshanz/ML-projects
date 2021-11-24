import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
sns.set()


def detect_winner(gameboard_in):
    row_checks = [gameboard_in[idx_list] for idx_list in [[val + (int(np.sqrt(len(gameboard_in))) * iter_)
                                                           for val in range(0, int(np.sqrt(len(gameboard_in))))]
                                                          for iter_ in range(int(np.sqrt(len(gameboard_in))))]]

    col_checks = [gameboard_in[idx_val] for idx_val in
                  [[val + iter_ for val in
                    range(0, int(np.sqrt(len(gameboard_in))) ** 2, int(np.sqrt(len(gameboard_in))))]
                   for iter_ in range(int(np.sqrt(len(gameboard_in))))]]
    forward_diag = [gameboard_in[val] for val in list(range(0, len(gameboard_in), int(np.sqrt(len(gameboard_in))) + 1))]
    backward_diag = [gameboard_in[val] for val in range(int(np.sqrt(len(gameboard_in))) - 1,
                                                        int(np.sqrt(len(gameboard_in))) ** 2 - int(
                                                            np.sqrt(len(gameboard_in)))+1,
                                                        int(np.sqrt(len(gameboard_in))) - 1)]

    all_check_lists = row_checks
    all_check_lists.extend(col_checks)
    all_check_lists.append(forward_diag)
    all_check_lists.append(backward_diag)

    winner = False
    for check_list in all_check_lists:
        if len(set(check_list)) == 1:
            if check_list[0] != 0:
                # print("Winner is:", check_list[0])
                return [True, check_list[0]]
    if not winner:
        return [False, 0]


def random_choice(gameboard_in, player_num):
    # Find all current taken spots on the gameboard
    chosen_idx = [idx for idx in range(len(gameboard_in)) if gameboard_in[idx] != 0]
    # Game is a draw
    if len(chosen_idx) == len(gameboard_in):
        return [gameboard_in, None]
    if len(chosen_idx) == 0:
        # Make first choice
        random_idx = random.randrange(0, len(gameboard_in))
        gameboard_in[random_idx] = player_num
    else:
        # Add a new choice
        random_idx = chosen_idx[0]
        while gameboard_in[random_idx] != 0:
            random_idx = random.randrange(0, len(gameboard_in))
        gameboard_in[random_idx] = player_num
    return [gameboard_in, random_idx]


def play_game(gameboard_length):
    # Storing the boards and decisions made with those boards for each player
    player_1_gameboards, player_1_choices = list(), list()
    player_2_gameboards, player_2_choices = list(), list()

    # Setting up the game
    gameboard = np.zeros(gameboard_length**2)
    winner, winning_player = False, 0
    player_counter = 0
    while not winner:
        # Save the current board
        current_player = (player_counter % 2) + 1
        if current_player == 1:
            player_1_gameboards.append(np.array(gameboard))
        else:
            player_2_gameboards.append(np.array(gameboard))
        # Player makes their choice
        gameboard, chosen_spot = random_choice(gameboard, current_player)
        # Save the resulting choice
        if current_player == 1 and chosen_spot is not None:
            player_1_choices.append(chosen_spot)
        elif current_player == 2 and chosen_spot is not None:
            player_2_choices.append(chosen_spot)
        if chosen_spot is None:
            # print("Draw")
            return [0, player_1_gameboards, player_1_choices, player_2_gameboards, player_2_choices]
        else:
            winner, winning_player = detect_winner(gameboard)
            player_counter += 1
    return [winning_player, player_1_gameboards, player_1_choices, player_2_gameboards, player_2_choices]


def generate_training_rows(num_games):
    player_1_x, player_1_y, player_2_x, player_2_y = list(), list(), list(), list()

    for game in range(num_games):
        game_output = play_game(gameboard_length=3)
        # Formatting the training rows
        if game % 2 == 0:
            for gameboard_data, win_data in zip(game_output[1], game_output[2]):
                player1_input_row = list(gameboard_data)
                player1_input_row.append(win_data)
                player_1_x.append(np.array(player1_input_row))
                if game_output[0] == 0:
                    player_1_y.append((0, 1, 0))
                elif game_output[0] == 1:
                    player_1_y.append((0, 0, 1))
                else:
                    player_1_y.append((1, 0, 0))

            for gameboard_data, win_data in zip(game_output[3], game_output[4]):
                player2_input_row = list(gameboard_data)
                player2_input_row.append(win_data)
                player_2_x.append(np.array(player2_input_row))
                if game_output[0] == 0:
                    player_2_y.append((0, 1, 0))
                elif game_output[0] == 1:
                    player_2_y.append((0, 0, 1))
                else:
                    player_2_y.append((1, 0, 0))
        else:
            for gameboard_data, win_data in zip(game_output[3], game_output[4]):
                player1_input_row = list(gameboard_data)
                player1_input_row.append(win_data)
                player_1_x.append(np.array(player1_input_row))
                if game_output[0] == 0:
                    player_1_y.append((0, 1, 0))
                elif game_output[0] == 1:
                    player_1_y.append((0, 0, 1))
                else:
                    player_1_y.append((1, 0, 0))

            for gameboard_data, win_data in zip(game_output[1], game_output[2]):
                player2_input_row = list(gameboard_data)
                player2_input_row.append(win_data)
                player_2_x.append(np.array(player2_input_row))
                if game_output[0] == 0:
                    player_2_y.append((0, 1, 0))
                elif game_output[0] == 1:
                    player_2_y.append((0, 0, 1))
                else:
                    player_2_y.append((1, 0, 0))


    return [player_1_x, player_1_y, player_2_x, player_2_y]


def train_neural_model(x_train, y_train, num_epochs_, batch_size_in_):
    # Initializing the mode
    model = Sequential()

    # Creating the learning architecture
    # model.add(Dense(32, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(64, input_shape=(x_train.shape[1], )))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(y_train[0])))
    model.add(Activation('softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=num_epochs_, batch_size=batch_size_in_, verbose=2, validation_split=.2)

    return model


def make_move_predict(gameboard_in, trained_model, player_num):
    open_idx = [idx for idx in range(len(gameboard_in)) if gameboard_in[idx] == 0]
    pred_rows = list()
    for idx in open_idx:
        pred_row = list(gameboard_in)
        pred_row.append(idx)
        pred_rows.append(pred_row)
    if len(pred_rows) > 0:
        best_choice = open_idx[int(np.argmax(trained_model.predict(np.array(np.array(pred_rows)))))]
        gameboard_in[best_choice] = player_num
        return [gameboard_in, best_choice]
    else:
        return [gameboard_in, None]


def train_forest_model(x_train, y_train):
    forest_model = RandomForestClassifier()
    forest_model.fit(x_train, y_train)
    return forest_model



