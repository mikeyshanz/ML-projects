import random

# Player 1 is 1, player 2 is 2, empty board spot is 0
size_vars = [
    (5, 4),
    (6, 5),
    (8, 7),
    (9, 7),
    (10, 7),
    (8, 8),
    (1000, 1000)
]


def generate_board(board_size):
    output = [[0 for row_ in range(board_size[0])] for col_ in range(board_size[1])]
    return output


def generate_random_move(board_in, player_in):
    # Pick a random, non-full column and drop a piece in
    empty_indicies = list()
    for x in range(len(board_in[0])):
        for y in range(len(board_in)):
            if board_in[y][x] not in [1, 2]:
                empty_indicies.append((x, y))

    if len(empty_indicies) == 0:
        return {'board': board_in, 'finished': True}

    avail_cols = list(set([empt[0] for empt in empty_indicies]))
    random_col = random.choice(avail_cols)
    chosen_spot = (random_col, -1)
    for empty_index in empty_indicies:
        if empty_index[0] == random_col:
            if empty_index[1] > chosen_spot[1]:
                chosen_spot = empty_index

    board_in[chosen_spot[1]][chosen_spot[0]] = player_in
    return {'board': board_in, 'finished': False}


def detect_horizontal_win(board_in):
    status = {'winner': False, 'winner_id': None}
    for row in board_in:
        in_row = 0
        for idx in range(1, len(row)):
            if row[idx] == row[idx-1] and row[idx] != 0:
                in_row += 1
                if in_row == 3:
                    status['winner'] = True
                    status['winner_id'] = row[idx]
                    return status
            else:
                in_row = 0
    return status


def detect_vertical_win(board_in):
    col_num = len(board_in[0])
    col_list = list()
    for col_idx in range(col_num):
        tmp_col = list()
        for row in board_in:
            tmp_col.append(row[col_idx])
        col_list.append(tmp_col)
    status = detect_horizontal_win(col_list)
    return status


def check_to_right_diagonal_win(board_in):
    all_rows = list()
    row = len(board_in)
    col = len(board_in[0])
    for line in range(1, (row + col)):
        start_col = max(0, line - row)
        count = min(line, (col - start_col), row)
        tmp_row = list()
        for j in range(0, count):
            tmp_row.append(board_in[min(row, line) - j - 1][start_col + j])
        all_rows.append(tmp_row)
    status = detect_horizontal_win(all_rows)
    return status


expected_y_path = [4,  # 1
                   4, 3,  # 2
                   4, 3, 2,  # 3
                   4, 3, 2, 1,  # 4
                   3, 2, 1, 0,  # 5
                   2, 1, 0,  # 6
                   1, 0,  # 7
                   0]  # 8

expected_x_path = [0,  # 1
                   1, 0,  # 2
                   2, 1, 0,  # 3
                   3, 2, 1, 0,  # 4
                   3, 2, 1, 0,  # 5
                   3, 2, 1,  # 6
                   3, 2,  # 7
                   3]  # 8

pairs = [(x, y) for x, y in zip(expected_x_path, expected_y_path)]


def check_to_left_diagonal_win(board_in):
    all_rows = list()
    row = len(board_in)
    col = len(board_in[0])
    y_array = list()
    for line in range(1, (row + col)):
        """
        row is always 5
        col is always 4
        line goes from 1 to 9 (8 total)
        
        y starts 4 4 4 4 3 2 1 0 - on 5th line it goes down
        x starts 0 1 2 3 3 3 3 3  - once it gets to 3 it stops
        """
        start_x = min((col - row + line), col - 1)
        start_y = min(col, (col - (line - col)))

        print(start_x, start_y)
        print()

        while start_x > -1:
            pair = (start_y, start_x)
            print(pair)
            start_x -= 1
            start_y -= 1


board_in = [
    [1,   2,   3,   4],
    [5,   6,   7,   8],
    [9,   10,  11,  12],
    [13,  14,  15,  16],
    [17,  18,  19,  20]
]


def detect_winner(board_in):
    status = detect_horizontal_win(board_in)
    if status['winner']:
        return status

    status = detect_vertical_win(board_in)
    if status['winner']:
        return status

    status = check_to_right_diagonal_win(board_in)
    if status['winner']:
        return status

    return status


def play_random_game():
    board_len = random.randrange(5, 20)
    board_height = random.randrange(5, 20)
    board = generate_board((board_len, board_height))
    horizontal_winner = detect_winner(board)
    player_iter = random.randrange(1, 3)
    board_finished = False
    while not board_finished:
        board_output = generate_random_move(board, player_iter)
        board_finished = board_output.get('finished')
        board = board_output['board']

        horizontal_winner = detect_winner(board)
        if horizontal_winner['winner']:
            # print(f"Winner is: {horizontal_winner['winner_id']}")
            return horizontal_winner
        if player_iter == 1:
            player_iter = 2
        else:
            player_iter = 1

    if not horizontal_winner['winner']:
        # print("Game Tied")
        return horizontal_winner


winner_list = list()

for _ in range(10):
    output = play_random_game()
    if output['winner_id'] is not None:
        winner_list.append(output['winner_id'])


# This should approach 1.5 for trueish random
print(sum(winner_list)/len(winner_list))

# Python3 program to print all elements
# of given matrix in diagonal order
# ROW = 5
# COL = 4


# Main function that prints given
# matrix in diagonal order

