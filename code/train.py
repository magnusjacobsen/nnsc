import sys, os, pickle, torch, time, math
import nnsc.savefile as sf
import nnsc.config as cf
import nnsc.network as net
import nnsc.gamestorage as gs
from nnsc import nnsc
import numpy as np
import torch.nn as nn
import hex.game as game

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
  # some default values
  params = {}
  save_mode = False
  save_version = ''
  load = False
  load_version = ''

  # checking the args from commandline
  for i in range(1, len(sys.argv)):
    s = sys.argv[i]
    if s.startswith('size='):
      params['size'] = int(s.split('size=')[1])
    elif s.startswith('sims='):
      params['num_simulations'] = int(s.split('=')[1])
    elif s.startswith('blocks='):
      params['num_res_blocks'] = int(s.split('=')[1])
    elif s.startswith('filters='):
      params['num_conv_filters'] = int(s.split('=')[1])
    elif s.startswith('actors='):
      params['num_actors'] = int(s.split('=')[1])
    elif s.startswith('epochs='):
      params['num_epochs'] = int(s.split('=')[1])
    elif s.startswith('window='):
      params['window_size'] = int(s.split('=')[1])
    elif s.startswith('batch='):
      params['batch_size'] = int(s.split('=')[1])
    elif s.startswith('singleprecision'):
      params['use_half_precision'] = False
    elif s.startswith('type='):
      tp = s.split('=')[1]
      if tp == 'u':
        params['use_utility'] = True
      elif tp == 'q':
        params['use_utility'] = True
        params['use_q_value'] = True
      elif tp == 'fpa':
        params['use_utility'] = True
        params['use_q_value'] = True
        params['use_full_pre_activation'] = True
      else:
        print('unknown type')
        exit(0)
    elif s.startswith('save='):
      save_mode = True
      save_version = s.split('=')[1]
    elif s.startswith('load='):
      load = True
      load_version = s.split('=')[1]
    else:
      print(f'argument {s} could not be parsed')
      exit(0)

  if load:
    savefile = sf.load_from_disk(load_version)
    # reflect saving arguments from from command line
    savefile.save_mode = save_mode
    savefile.version = save_version
    print('**********************')
    if not hasattr(savefile.config, 'use_half_precision'):
      savefile.config.use_half_precision = True

  else: # create new config and savefile objects
    config = cf.Config(**params)
    savefile = sf.Savefile(config, save_mode = save_mode, version = save_version)

  # create folder if load and save version does not match, and saving is active
  if save_mode and save_version != load_version:
    sf.make_folder(savefile)
  
  nnsc.run_nnsc(savefile)