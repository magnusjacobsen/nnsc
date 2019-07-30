import numpy as np
import math, torch
import numba as nb

# (Game) State class:
# - board: int8 array, with dim size x size. 0 indicates an empty place, 1 player 1 piece, -1 player 2 piece.
#          actually flattened to 1D. I thought it would be quicker.
# - uf: union-find 1D array, with length size * size + 4. The last 4 indices are 2 for player 1's winning sides, the other for player 2.
#       if a player has connection between the two, the game is terminal, in that player's favor.
# - actions: set of possible actions. Starts as all the numbers of 0 through size * size - 1. Every time an action is taken (def result), the corresponding
#       action int is removed from this set. It ensures linear time in calculating legal actions from a state (linear because we convert set to list).
# - player_to_play: the player that is about to take a turn. 1 for player 1, -1 for player 2. Easy to change player, just add minus in front of current player value.
# - size: length of one of the sides of the board
# - history: list of all (integer) actions taken, in chronological order.
# - is_terminated: boolean, saying if game is a terminal state.
class State:
  def __init__(self, board, uf, actions, player_to_play = 1, size = 7, history = [], is_terminated = False):
    self.history = history
    self.player_to_play = player_to_play
    self.board = board
    self.uf = uf
    self.size = size
    self.actions = actions
    self.is_terminated = is_terminated
    self.num_positions = size * size

  # Complete copy of the current game state
  def copy(self):
    return State(self.board.copy(),
                 self.uf.copy(),
                 self.actions.copy(),
                 history = self.history.copy(),
                 is_terminated = self.is_terminated,
                 size = self.size,
                 player_to_play = self.player_to_play)

# we are limiting our board to a max side size of 256
# a board have values:
# 1, for player 1
# 0, for free
# -1, for player 2 (or opponent, or player -1)
def initial_state(size):
  board = np.zeros(size * size, dtype = np.int8)
  uf = np.arange(0, (size * size) + 4, dtype = np.int16)
  actions = set(range(size * size))
  return State(board, uf, actions, size = size)

# returns the set of all legal moves in a state (for the given player)
def actions(state):
  if not terminal_test(state):
    return list(state.actions)
  else: return []

# defines which player has the move in a state
# player 1 returns a 1, player 2 returns a -1.
def player(state):
  return state.player_to_play

# objective function or payoff function. Defines the final numerical value for a game that ends in a terminal state s for player p.
# here:
# -  1 = win
# -  0 = not finished (there are no draws in hex)
# - -1 = loss
def utility(state, player):
  return _utility(state.num_positions, state.uf, player)

# returns true if the game is over (if one of the players has won or lost),
# false if not
def terminal_test(state):
  return state.is_terminated

# transitional model, returns the resulting state from a given action in a given state
def result(state, action):
  if not (state.board[action] == 0): return state # action not possible
  state = state.copy()
  # board update
  state.board[action] = state.player_to_play
  # union-find update
  _update_union_find(action, state.size, state.num_positions, state.player_to_play, state.board, state.uf)
  # player update (just flip the value of the former player)
  state.player_to_play = -state.player_to_play
  # actions update
  state.actions.remove(action)
  state.history.append(action)
  # is_terminated update (if the current player has either lost or won, the game is over):
  state.is_terminated = not (_utility(state.num_positions, state.uf, state.player_to_play) == 0)
  return state

# Includes 2 x 2D binary feature plane arrays
# - 1st array current players piecess (1 for a piece, 0 for none)
# - 2nd array opponents players pieces (1 for a piece, 0 for none)
#
# If the turn, at the requested step, is player 2's turn,
# we transpose both boards, such that current player always have same perspective
@nb.jit
def _image_stack(history, size, step):
  img = np.zeros((2, size, size), dtype = np.float32)
  step = len(history) if step == -1 else step
  playerOnePerspective = step % 2 == 0 # it is player 1s turn to take an ation
  # maybe this is cheaper that transposing the result from one's perspective
  for i_step, action in enumerate(history[:step]):
    if playerOnePerspective:
      i, j = action // size, action % size
      player = i_step % 2 # even steps: 0, odd: 1
    else:
      i, j = action % size, action // size # transposed tables, to make the perspective uniform for the current player, who's turn it is at step
      player = (i_step - 1) % 2 # odd steps = 0, even: 1
    img[player][i][j] = 1
  return img

@nb.jit
def _utility(num_pos, uf, player):
  if _connected(num_pos, num_pos + 1, uf): # one's two goal sides in union find array
    return player
  if _connected(num_pos + 2, num_pos + 3, uf): # two's two goal sides
    return -player
  return 0

# - Returns all possible connections from a given position, taking the size into account
# - used in _update_union_find()
# - This is also where it's apparent that the hex game uses a 2D matrix/array as a hexagonal grid,
#   by only allowing certain move directions (when connecting one's sides)
# - The effect of this is that we "tilt" the board, and end up with 6 connecting sides, thus a hexagonal grid
# - Illustration (c: current index, p: possible connection):
#    [ ] [ ] [ ] [ ] [ ]
#    [ ] [p] [p] [ ] [ ]
#    [ ] [p] [C] [p] [ ]
#    [ ] [ ] [p] [p] [ ]
#    [ ] [ ] [ ] [ ] [ ]
@nb.jit
def _possible_connections(index, size):
  row, col = index // size, index % size
  pcon = [] # list of possible connections
  if row > 0:
    pcon.append(index - size) # top
    if col > 0: pcon.append(index - size - 1) # topleft
  if row < (size - 1):
    pcon.append(index + size) # bottom
    if col < (size - 1) : pcon.append(index + size + 1) # bottomright
  if col > 0: pcon.append(index - 1) # left
  if col < (size - 1): pcon.append(index + 1) # right
  return pcon

@nb.jit
def _update_union_find(action, size, num_pos, player, board, uf):
  for i in _possible_connections(action, size):
    if board[i] == player:
      _union(action, i, uf)
  # if border tile, check if it belongs to current player
  if player == 1:
    row = action // size
    if row == 0: _union(action, num_pos, uf)                # player one's first goal side
    elif row == size - 1: _union(action, num_pos + 1, uf)   # player one's second goal side
  else:
    column = action % size
    if column == 0: _union(action, num_pos + 2, uf)         # player two's first goal side
    elif column == size - 1:_union(action, num_pos + 3, uf) # player two's second goal side

#########################
# Quick-Union algorithm #
#  Sedgewick & Wayne's  #
#########################
@nb.jit
def _connected(p: int, q: int, arr):
  return _find(p, arr) == _find(q, arr)

@nb.jit
def _find(p: int, arr):
  while not (p == arr[p]): p = arr[p]
  return p

@nb.jit
def _union(p: int, q: int, arr):
  i = _find(p, arr)
  j = _find(q, arr)
  if not (i == j): arr[i] = j