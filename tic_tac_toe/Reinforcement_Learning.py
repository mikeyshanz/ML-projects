from tac_game import *
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

# Training each player with their input data
training_output = generate_training_rows(100)
x_1, y_1 = np.array(training_output[0]), np.array(training_output[1])
x_2, y_2 = np.array(training_output[2]), np.array(training_output[3])
del training_output  # reduce the load on memory
# Shuffling the training data
X_1_sparse, X_2_sparse = coo_matrix(x_1), coo_matrix(x_2)
x_1, X_1_sparse, y_1 = shuffle(x_1, X_1_sparse, y_1, random_state=0)
x_2, X_2_sparse, y_2 = shuffle(x_2, X_2_sparse, y_2, random_state=0)

# Training the models
player_1_neural_model = train_neural_model(x_train=x_1, y_train=y_1, num_epochs_=10, batch_size_in_=128)
player_2_neural_model = train_neural_model(x_train=x_2, y_train=y_2, num_epochs_=10, batch_size_in_=128)
player_1_forest_model = train_forest_model(x_train=x_1, y_train=y_1)
player_2_forest_model = train_forest_model(x_train=x_2, y_train=y_2)

# Getting new data for more validation
# Training each player with their input data
new_training_output = generate_training_rows(100)
new_x_1, new_y_1 = np.array(new_training_output[0]), np.array(new_training_output[1])
new_x_2, new_y_2 = np.array(new_training_output[2]), np.array(new_training_output[3])
# Shuffling the training data
new_X_1_sparse, new_X_2_sparse = coo_matrix(new_x_1), coo_matrix(new_x_2)
new_x_1, new_X_1_sparse, new_y_1 = shuffle(new_x_1, new_X_1_sparse, new_y_1, random_state=0)
new_x_2, new_X_2_sparse, new_y_2 = shuffle(new_x_2, new_X_2_sparse, new_y_2, random_state=0)

# Making the predictions
neural_1_preds = player_1_neural_model.predict_classes(new_x_1)
neural_2_preds = player_2_neural_model.predict_classes(new_x_2)
forest_1_preds = [np.argmax(output_row) for output_row in player_1_forest_model.predict(new_x_1)]
forest_2_preds = [np.argmax(output_row) for output_row in player_2_forest_model.predict(new_x_2)]

# Converting the y's to class predictions
new_y_1_classes = [np.argmax(input_row) for input_row in new_y_1]
new_y_2_classes = [np.argmax(input_row) for input_row in new_y_2]

# Getting the scores
print("Neural 1 Validation:", r2_score(y_true=new_y_1_classes, y_pred=neural_1_preds))
print("Neural 2 Validation:", r2_score(y_true=new_y_2_classes, y_pred=neural_2_preds))
print("Forest 1 Validation:", r2_score(y_true=new_y_1_classes, y_pred=forest_1_preds))
print("Forest 2 Validation:", r2_score(y_true=new_y_2_classes, y_pred=forest_2_preds))

plt.figure()
plt.plot(new_y_2_classes, label='Actual')
plt.plot(neural_2_preds, label='Preds')
plt.legend()
plt.show()

#
# # Now having each model play against random guesses
# player_1_gameboards, player_1_choices = list(), list()
# player_2_gameboards, player_2_choices = list(), list()
#
# # Setting up the game
# winner_list = list()
# for play_ in range(5):
#     gameboard_length = 3
#     gameboard = np.zeros(gameboard_length ** 2)
#     winner, winning_player = False, 0
#     player_counter = 0
#     while not winner:
#         # Save the current board
#         current_player = (player_counter % 2) + 1
#         if current_player == 1:
#             player_1_gameboards.append(np.array(gameboard))
#             gameboard, chosen_spot = make_move_predict(gameboard, player_2_model, 1)
#         else:
#             player_2_gameboards.append(np.array(gameboard))
#             gameboard, chosen_spot = make_move_predict(gameboard, player_1_model, 2)
#
#         # Save the resulting choice
#         if current_player == 1 and chosen_spot is not None:
#             player_1_choices.append(chosen_spot)
#         elif current_player == 2 and chosen_spot is not None:
#             player_2_choices.append(chosen_spot)
#         if chosen_spot is None:
#             # print("Draw")
#             winner = True
#             # return [0, player_1_gameboards, player_1_choices, player_2_gameboards, player_2_choices]
#         else:
#             winner, winning_player = detect_winner(gameboard)
#             player_counter += 1
#     winner_list.append(winning_player)
#     print(gameboard.reshape(3, 3))
#
# # return [winning_player, player_1_gameboards, player_1_choices, player_2_gameboards, player_2_choices]
# win_1, win_2, draw = 0, 0, 0
# for win in winner_list:
#     if win == 0:
#         draw += 1
#     elif win == 1:
#         win_1 += 1
#     else:
#         win_2 += 1
#
# print("Draws:", draw)
# print("Win 1:", win_1)
# print("Win 2:", win_2)
