import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# convulation layer, with 3x3 kernel size, with padding = 1
def conv3x3(in_planes, out_planes, stride = 1):
  return nn.Conv2d(in_planes, out_planes, kernel_size = 3, padding = 1, stride = stride, bias = False)

# needed for value head!
def conv1x1(in_planes, out_planes, stride = 1):
  return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)

def make_res_blocks(config, planes, activation, stride, num_res_blocks):
  blocks = [ResBlock(config, planes, activation, stride) for x in range(0, num_res_blocks)]
  return nn.Sequential(*blocks)

# the residual learning block
class ResBlock(nn.Module):
  def __init__(self, use_full_pre_activation, planes, activation, stride):
    super(ResBlock, self).__init__()
    self.conv1 = conv3x3(planes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.activation = activation
    self.conv2 = conv3x3(planes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)
    self.stride = stride
    self.use_full_pre_activation = use_full_pre_activation # full pre-activation
    
  # Each block consists of:
  # - 2 rectified, batch-normalized convolutional layers, with a skip connection
  # - each conv applies 256 filters, kernel size 3x3, and stride = 1
  # full pre-actiovation block
  # https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e
  def forward(self, input):
    if self.use_full_pre_activation: # full pre-activation layout
      identity = input 
      output = self.bn1(input)
      output = self.activation(output)
      output = self.conv1(output)
      output = self.bn2(output)
      output = self.activation(output)
      output = self.conv2(output)
      output = output + identity
      return output
    else: # standard residual block layout. Same as in AlphaZero paper
      identity = input
      output = self.conv1(input)
      output = self.bn1(output)
      output = self.activation(output)
      output = self.conv2(output)
      output = self.bn2(output)
      output = output + identity
      output = self.activation(output)
      return output

# now it gets exciting!!
class Network(nn.Module):
  def __init__(self, config):
    super(Network, self).__init__()
    self.size = config.size
    self.planes = config.num_conv_filters
    action_size = config.action_size # size * size
    stride = config.conv_stride
    num_res_blocks = config.num_res_blocks

    # foot and body
    self.relu = nn.ReLU(inplace = True)
    self.foot_conv = conv3x3(2, self.planes, stride) # two channels in, the two board images
    self.foot_bn = nn.BatchNorm2d(self.planes)
    self.body = make_res_blocks(config.use_full_pre_activation, self.planes, self.relu, stride, num_res_blocks)

    self.policy_conv = conv3x3(self.planes, self.planes)
    self.policy_bn = nn.BatchNorm2d(self.planes)
    self.policy_fc = nn.Linear(self.planes * self.size * self.size, action_size) # out: batch_size x board_size * board_size

    self.value_conv = conv1x1(self.planes, 1)
    self.value_bn = nn.BatchNorm2d(1)
    self.value_fc1 = nn.Linear(self.size * self.size, self.planes)
    self.value_fc2 = nn.Linear(self.planes, 1)
    self.value_tanh = nn.Tanh()

  # following the description from Alpha Zero paper
  def forward(self, x):
    # rectified, batch-normalized, conv layer
    x = self.foot_conv(x)
    x = self.foot_bn(x)
    x = self.relu(x)
    # 19 residual blocks
    x = self.body(x)
    # policy head:
    # 1. rectified, batch-normalized conv layer
    # 2. linear layer with output size = action size (representing the logits)
    policy = self.policy_conv(x)
    policy = self.policy_bn(policy)
    policy = self.relu(policy)
    policy = policy.view(-1, self.planes * self.size * self.size) # flatten the input: batch_size x planes * board_size * board_size
    policy = self.policy_fc(policy)
    # value head:
    # 1. rectified, batch-normalized conv of filter 1, kernel 1, stride 1
    # 2. rectified linear layer of size 256
    # 3. tanh-linear layer of 1 size
    value = self.value_conv(x)      # 1.
    value = self.value_bn(value)    # 1.
    value = self.relu(value)        # 1.
    value = value.view(-1, self.size * self.size) #flatten to size batch_size x board_size * board_size
    value = self.value_fc1(value)   # 2.
    value = self.relu(value)        # 2.
    value = self.value_fc2(value)   # 3.
    value = self.value_tanh(value)  # 3.
    return value, policy

def inference(model : Network, input, players, size : int):
  with torch.no_grad():
    values, policies = model(input)
    values = values.cpu().numpy()
    policies = policies.cpu().numpy()
    policies = [correct_perspective(pol, plr, size) for pol, plr in zip(policies, players)]
    return values, policies

# if player is -1 (player two)
# does in-place matrix transposition of policies
# of a 2D square matrix flattened to 1D
def correct_perspective(policy, player, size):
  if player == -1:
    for i in range(0, size - 1):
      for j in range(i + 1, size):
        one = i * size + j
        one_t = j * size + i
        temp = policy[one]
        policy[one] = policy[one_t]
        policy[one_t] = temp
  return policy

# creates a copy of a float32 network, and changes most layers to float16
# except for batch norm layers. https://discuss.pytorch.org/t/training-with-half-precision/11815
def copy16(model, config):
  copy = Network(config)
  copy.load_state_dict(model.state_dict())
  copy = copy.to(DEVICE)
  if str(DEVICE).startswith('cuda'):
    #copy = copy.half()
    for layer in copy.modules():
      if isinstance(layer, nn.BatchNorm2d): 
        layer.float()
      if (isinstance(layer, nn.Conv2d) or
        isinstance(layer, nn.Conv1d) or
        isinstance(layer, nn.Linear) or
        isinstance(layer, nn.ReLU) or
        isinstance(layer, nn.Tanh)):
        layer.half()
  return copy

def make_tensors(image_stacks, config):
  tensors = torch.from_numpy(np.concatenate(image_stacks, axis = 0))
  tensors = tensors.to(DEVICE)

  if config.use_half_precision:
    tensors = tensors.half()
  else:
    tensors = tensors.float()
    
  return tensors

###########################
#### Training helpers #####
###########################
# copied from https://github.com/sotte/pytorch_tutorial/blob/master/notebooks/03_transfer_learning.ipynb
def get_trainable(model_params):
  return (p for p in model_params if p.requires_grad)

def get_frozen(model_params):
  return (p for p in model_params if not p.requires_grad)

def all_trainable(model_params):
  return all(p.requires_grad for p in model_params) # returns boolean

def all_frozen(model_params):
  return all(not p.requires_grad for p in model_params) # returns a boolean

def freeze_all(model_params):
  for p in model_params: p.requires_grad = False

def unfreeze_all(model_params):
  for p in model_params: p.requires_grad = True
