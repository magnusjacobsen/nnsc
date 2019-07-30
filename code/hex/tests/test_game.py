import pytest
import numpy as np
import hex.game as game

# testing that actions(s) returns the correct actions
def test_actions_correct_value():
  initial = game.initial_state(7)
  actual = game.actions(initial)
  actual.sort()
  expected = list(range(49)) # 7 * 7
  assert actual == expected

# testing that actions(s) returns the correct amount of actions
def test_actions_correct_count():
  initial = game.initial_state(8)
  state = game.result(initial, 1)
  actions = game.actions(state)
  expected = 63 # 8 * 8 - 1
  actual = len(actions)
  assert actual == expected

# testing that the state has the correct starting player
def test_player_one_starts():
  state = game.initial_state(0)
  expected = 1
  actual = game.player(state)
  assert actual == expected

# testing that result changes the player of the state
def test_result_player_change():
  initial = game.initial_state(1)
  state = game.result(initial, 0)
  expected = -1
  actual = game.player(state)
  assert actual == expected

# testing that a copy of a state has identical board
def test_copy_game_board_equals():
  initial = game.initial_state(11)
  state = game.result(game.result(game.result(initial, 0), 1), 2)
  copy = state.copy()
  actual = copy.board
  expected = state.board
  assert np.array_equal(actual, expected)

# testing that copying a state doesn't result in a shallow copy, where later actions affect the original board state
def test_copy_result_board_not_equals():
  initial = game.initial_state(11)
  state = game.result(game.result(game.result(initial, 0), 1), 2)
  copy = state.copy()
  copy = game.result(copy, 3)
  expected = state.board
  actual = copy.board
  assert not np.array_equal(actual, expected)

# test that utility returns 0 when the game is not finished
def test_utility_not_terminal_player_one():
  initial = game.initial_state(6)
  actual = game.utility(initial, 1)
  expected = 0
  assert actual == expected

# test that utility returns 0 when the game is not finished
def test_utility_not_terminal_player_two():
  initial = game.initial_state(6)
  actual = game.utility(initial, -1)
  expected = 0
  assert actual == expected

# returns correct value, -1, when player 1 (1) lost
def test_utility_player_one_lost():
  initial = game.initial_state(2)
  fst = game.result(initial, 3)
  snd = game.result(fst, 0)
  thd = game.result(snd, 2)
  fth = game.result(thd, 1)
  actual = game.utility(fth, 1)
  expected = -1
  assert actual == expected

# returns correct value, 1, when player 1 (1) won
def test_utility_player_one_won():
  initial = game.initial_state(2)
  fst = game.result(initial, 0)
  snd = game.result(fst, 1)
  thd = game.result(snd, 2)
  actual = game.utility(thd, 1)
  expected = 1
  assert actual == expected

# returns correct value, 1, when player 2 (-1) won
def test_utility_player_two_won():
  initial = game.initial_state(2)
  fst = game.result(initial, 3)
  snd = game.result(fst, 0)
  thd = game.result(snd, 2)
  fth = game.result(thd, 1)
  actual = game.utility(fth, -1)
  expected = 1
  assert actual == expected

# returns correct value, -1, when player 2 (-1) lost
def test_utility_player_two_lost():
  initial = game.initial_state(2)
  fst = game.result(initial, 0)
  snd = game.result(fst, 1)
  thd = game.result(snd, 2)
  actual = game.utility(thd, -1)
  expected = -1
  assert actual == expected

# result(s, a) changes player to opponent
def test_result_changes_player():
  initial = game.initial_state(3)
  state = game.result(initial, 0)
  actual = state.player_to_play
  expected = -1
  assert actual == expected

# result(s, a) adds action to returned states board array when player 1 takes an action
def test_result_adds_action_for_player_one():
  initial = game.initial_state(4)
  state = game.result(initial, 1)
  actual = state.board[1]
  expected = 1
  assert actual == expected

# result(s, a) adds action to returned states board array when player 2 takes an action
def test_result_adds_action_for_player_two():
  initial = game.initial_state(4)
  state = game.result(initial, 1)
  state = game.result(state, 2)
  actual = state.board[2]
  expected = -1
  assert actual == expected

# result(s,a) removes correct action from actions
def test_result_removes_correct_action():
  initial = game.initial_state(19)
  action = 55
  state = game.result(initial, action)
  actions = set(state.actions)
  result = action not in actions
  assert result

# result(s, a) appends correct action to history
def test_result_appends_history():
  initial = game.initial_state(33)
  action_one = 17
  action_two = 22
  state = game.result(initial, action_one)
  state = game.result(state, action_two)
  expected = [action_one, action_two]
  actual = state.history
  assert expected == actual

# result(s, a) updates union find data structure for player 1, where it is connected to a winning side
# uf[num positions]     = player one's first side
# uf[num positions + 1] = player one's second side
# uf[num positions + 2] = player two's first side
# uf[num positions + 3] = player two's second side
def test_result_updates_uf_player_one_side_connected():
  initial = game.initial_state(5)
  p1_side1 = 5 * 5
  action = 0
  state = game.result(initial, action)
  result = game._connected(p1_side1, action, state.uf)
  assert result

# result updates uf where not connected to winning side for player 1
def test_result_updates_uf_player_one_sides_not_connected():
  initial = game.initial_state(4)
  p1_side1 = 5 * 5
  action = 5
  state = game.result(initial, action)
  result = game._connected(p1_side1, action, state.uf)
  assert not result

# result correct updates union find where player two has connected to a winning side
def test_result_updates_uf_player_two_side_connected():
  initial = game.initial_state(5)
  p2_side1 = 5 * 5 + 2
  state = game.result(initial, 1)
  state = game.result(state, 0)
  result = game._connected(p2_side1, 0, state.uf)
  assert result

# result correct updates union find where player two has connected to a winning side
def test_result_updates_uf_player_two_side_not_connected():
  initial = game.initial_state(5)
  p2_side1 = 5 * 5 + 2
  state = game.result(initial, 1)
  state = game.result(state, 2)
  result = game._connected(p2_side1, 2, state.uf)
  assert not result

# _image_stack(h, size, step) returns correct np array when player one has to take a turn
# after player 2 has taken an action, the state will be from player 1's perspective
# which is "normal"
def test_image_stack_player_one_perspective():
  size = 3
  action_one = 0
  action_two = 1
  initial = game.initial_state(size)
  state = game.result(initial, action_one)
  state = game.result(state, action_two)
  expected = np.zeros((2, size, size))
  expected[0][action_one // size][action_one % size] = 1 # player one's action
  expected[1][action_two // size][action_two % size] = 1 # player two's action
  actual = game._image_stack(state.history, size, -1)
  assert np.array_equal(actual, expected)

# _image_stack(h, size, step) returns correct np array when player two has to take a turn
# after player 1 has taken an action, the state will be from player 2's perspective
# where the boards are transposed
def test_image_stack_player_two_perspective():
  size = 3
  action = 1
  initial = game.initial_state(size)
  state = game.result(initial, action)
  expected = np.zeros((2, size, size), dtype = np.float32)
  # player ones's action
  # notice we now use modulo first, and division second.
  # this results in a transposed matrix
  expected[1][action % size][action // size] = 1
  actual = game._image_stack(state.history, size, 1)
  assert np.array_equal(actual, expected)

# image_stack returns array with only 0-values, when step 0 is given, 
# even though the state has an action history beyond that 
def test_image_stack_with_0_step():
  size = 77
  initial = game.initial_state(size)
  state = game.result(initial, 55)
  state = game.result(state, 94)
  expected = np.zeros((2, size, size))
  actual = game._image_stack(state.history, size, 0)
  assert np.array_equal(actual, expected)

# _possible_connections gives correct values when in the top left corner
# remember that board is a 1D array
# which we conceptually can think of as a 2D matrix
# which again can be thought of as a hexagonal grid, 
# when we "tilt" the array, by adding some restrictions on connections
# - Illustration (c: current index, p: possible connection):
#    [ ] [ ] [ ] [ ] [ ]
#    [ ] [p] [p] [ ] [ ]
#    [ ] [p] [C] [p] [ ]
#    [ ] [ ] [p] [p] [ ]
#    [ ] [ ] [ ] [ ] [ ]
# in this case we have
#    [C] [p] [ ]
#    [p] [p] [ ]
#    [ ] [ ] [ ]
def test_possible_connections_top_left_corner():
  size = 11
  corner_index = 0
  actual = game._possible_connections(corner_index, size)
  actual.sort()
  expected = [corner_index + 1, 
              corner_index + size,
              corner_index + size + 1]
  assert actual == expected

# possible connection gives correct values for top right corner
#    [ ] [p] [C]
#    [ ] [ ] [p]
#    [ ] [ ] [ ]
def test_possible_connections_top_right_corner():
  size = 12
  corner_index = 11
  actual = game._possible_connections(corner_index, size)
  actual.sort()
  expected = [corner_index - 1,
              corner_index + size]
  assert actual == expected

# possible connection gives correct values for bottom left corner
#    [ ] [ ] [ ]
#    [p] [ ] [ ]
#    [C] [p] [ ]
def test_possible_connections_bottom_left_corner():
  size = 13
  corner_index = size * (size - 1)
  actual = game._possible_connections(corner_index, size)
  actual.sort()
  expected = [corner_index - size,
              corner_index + 1]
  assert actual == expected

# possible connection gives correct values for bottom right corner
#    [ ] [ ] [ ]
#    [ ] [p] [p]
#    [ ] [p] [C]
def test_possible_connections_bottom_right_corner():
  size = 7
  corner_index = size * size - 1
  actual = game._possible_connections(corner_index, size)
  actual.sort()
  expected = [corner_index - size - 1,
              corner_index - size,
              corner_index - 1]
  assert actual == expected

# test possible connection, in the middle, of a larger array, returns all six options
#    [ ] [ ] [ ] [ ] [ ]
#    [ ] [p] [p] [ ] [ ]
#    [ ] [p] [C] [p] [ ]
#    [ ] [ ] [p] [p] [ ]
#    [ ] [ ] [ ] [ ] [ ]
def test_possible_connections_center():
  size = 5
  center_index = 12
  actual = game._possible_connections(center_index, size)
  actual.sort()
  expected = [center_index - size - 1,
              center_index - size,
              center_index - 1,
              center_index + 1,
              center_index + size,
              center_index + size + 1]
  assert actual == expected

# test when index is on left side of array
#    [ ] [ ] [ ]
#    [p] [ ] [ ]
#    [C] [p] [ ]
#    [p] [p] [ ]
#    [ ] [ ] [ ]
def test_possible_connections_left_side():
  size = 45
  side_index = size * 7
  actual = game._possible_connections(side_index, size)
  actual.sort()
  expected = [side_index - size,
              side_index + 1,
              side_index + size,
              side_index + size + 1]
  assert actual == expected

# test when index is on top side of array
#    [ ] [p] [C] [p] [ ]
#    [ ] [ ] [p] [p] [ ]
#    [ ] [ ] [ ] [ ] [ ]
def test_possible_connections_top_side():
  size = 23
  side_index = size // 2
  actual = game._possible_connections(side_index, size)
  actual.sort()
  expected = [side_index - 1,
              side_index + 1,
              side_index + size,
              side_index + size + 1]
  assert actual == expected

# test when index is on right side of array
#    [ ] [ ] [ ]
#    [ ] [p] [p]
#    [ ] [p] [C]
#    [ ] [ ] [p]
#    [ ] [ ] [ ]
def test_possible_connections_right_side():
  size = 6
  side_index = size * 2 - 1 
  actual = game._possible_connections(side_index, size)
  actual.sort()
  expected = [side_index - size - 1,
              side_index - size,
              side_index - 1,
              side_index + size]
  assert actual == expected

# test when index is on bottom side of array
#    [ ] [ ] [ ] [ ] [ ]
#    [ ] [p] [p] [ ] [ ]
#    [ ] [p] [C] [p] [ ]
def test_possible_connections_bottom_side():
  size = 16
  side_index = size * size - size // 2
  actual = game._possible_connections(side_index, size)
  actual.sort()
  expected = [side_index - size - 1,
              side_index - size,
              side_index - 1,
              side_index + 1]
  assert actual == expected

###########################
# the union find algorithm functions have been tested by some of the result(s,a) tests
###########################