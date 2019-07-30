import sys, os, pickle, torch
import nnsc.network as net
import nnsc.gamestorage as gs
import nnsc.config as cf
import nnsc.savefile as sf
from nnsc import nnsc
import controller as ctrl
import hex.gui as gui
import ais.mcts as mcts
import ais.random_ai as random_ai
import datetime

def get_agent(s):
  if s.startswith('mcts'):
    if '!' in s:
      splitted = s.split('!')
      if len(splitted) > 2:
        return mcts.AI(num_iterations = int(splitted[1]), exploration_parameter = int(splitted[2]))
      else: 
        return mcts.AI(num_iterations = int(s.split('!')[1]))
    else:        return mcts.AI()

  elif s.startswith('random'):
    return random_ai.AI()

  elif s.startswith('human'):
    return None

  elif s.startswith('nn'):
    mp_params = s.split('!')
    version = mp_params[1]
    path = sf.PATH + version + '/'
    files = [f for f in os.listdir(path) if '.save' in f]
    play_one = True
    if len(mp_params) > 2:
      if mp_params[2] == 'all' or '-' in mp_params[2]: play_one = False
      else: files = [f for f in files if ('epoch' + mp_params[2] + '.save') in f]
    if len(files) < 1:
      print(f'nothing to load from directory {version}')
      exit(0)

    if play_one:
      files.sort(reverse = True)
      return get_nn_agent(path + files[0])

    else: # play all
      files.sort(reverse = True)
      if '-' in mp_params[2]:
        from_epoch = mp_params[2].split('-')[0]
        to_epoch = mp_params[2].split('-')[1]
        from_index, to_index = None, None
        for i, f in enumerate(files):
          epoch = f.split('epoch')[1].split('.')[0]
          if epoch == from_epoch: from_index = i
          if epoch == to_epoch: to_index = i
        if from_index < to_index:
          files = files[from_index : to_index + 1]
        else:
          files = files[to_index : from_index + 1]
      return [path + f for f in files]
  else: return None

def get_nn_agent(path):
  version = path.split('saves/')[1].split('/')[0]
  with open(path, 'rb') as f:
    savefile = pickle.load(f)
    print('**********************')
    if not hasattr(savefile.config, 'use_half_precision'):
      savefile.config.use_half_precision = True
    print(path)
  return nnsc.AI(savefile)

if __name__ == '__main__':
  if len(sys.argv) > 6 or len(sys.argv) < 5:
    print('too few or too many arguments')
    exit(0)
  agent1 = get_agent(sys.argv[1])
  agent2 = get_agent(sys.argv[2])
  size = int(sys.argv[3])
  games_to_play = int(sys.argv[4])
  visual = False
  if len(sys.argv) > 5 and sys.argv[5] == 'vis':
    visual = True
  
  # if agent1 or agent2 are human they are None, we set the board to visible, max the games at 2 (one as either player)
  if (not agent1) or (not agent2):
    if isinstance(agent1, list):
      print("Playing against multiple AI opponents is not supported")
      exit(0)
    games_to_play = min(2, games_to_play)
    visual = True
  
  print(f'board_size: {size}x{size}')
  controller = ctrl.Controller(agent1, agent2, size, visual = visual)
  if isinstance(agent1, list):
    controller.run_many(agent1, agent2, games_to_play)
  else:
    controller.run(games_to_play)