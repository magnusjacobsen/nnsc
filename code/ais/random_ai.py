import hex.game as game
import random as random

class AI():
  def decide_move(self, state):
    return random_move(state)

  def decide_parallel(self, states):
    actions = [self.decide_move(state) for state in states]
    return actions

  def __str__(self):
    return 'random_ai'
  def __repr__(self):
    return 'random_ai.AI()'

def random_move(state):
  actions = game.actions(state)
  action = actions[random.randrange(0, len(actions))] # instead of randint, which would potentially also give the len, which is outside of array range
  return action