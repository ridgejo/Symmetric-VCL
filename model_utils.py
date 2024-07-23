import numpy as np
import torch
from scipy.stats import truncnorm

''' Random coreset selection '''
def rand_from_batch(x_coreset, y_coreset, x_train, y_train, coreset_size):
  # Randomly select from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
  idx = np.random.choice(x_train.shape[0], coreset_size, False)
  x_coreset.append(x_train[idx,:])
  y_coreset.append(y_train[idx])
  x_train = np.delete(x_train, idx, axis=0)
  y_train = np.delete(y_train, idx, axis=0)
  return x_coreset, y_coreset, x_train, y_train

''' K-center coreset selection '''
def k_center(x_coreset, y_coreset, x_train, y_train, coreset_size):
  # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
  dists = np.full(x_train.shape[0], np.inf)
  current_id = 0
  dists = update_distance(dists, x_train, current_id)
  idx = [ current_id ]

  for i in range(1, coreset_size):
      current_id = np.argmax(dists)
      dists = update_distance(dists, x_train, current_id)
      idx.append(current_id)

  x_coreset.append(x_train[idx,:])
  y_coreset.append(y_train[idx])
  x_train = np.delete(x_train, idx, axis=0)
  y_train = np.delete(y_train, idx, axis=0)

  return x_coreset, y_coreset, x_train, y_train

def update_distance(dists, x_train, current_id):
  for i in range(x_train.shape[0]):
    current_dist = np.linalg.norm(x_train[i,:]-x_train[current_id,:])
    dists[i] = np.minimum(current_dist, dists[i])
  return dists

# variable initialization functions
def truncated_normal(size, stddev=1, variable = False, mean=0):
  mu, sigma = mean, stddev
  lower, upper= -2 * sigma, 2 * sigma
  X = truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
  X_tensor = torch.Tensor(data = X.rvs(size)).to(device = device)
  X_tensor.requires_grad = variable
  return X_tensor

def init_tensor(value,  dout, din = 1, variable = False):
  if din != 1:
    x = value * torch.ones([din, dout]).to(device = device)
  else:
    x = value * torch.ones([dout]).to(device = device)
  x.requires_grad=variable

  return x

def merge_coresets(x_coresets, y_coresets):
  merged_x, merged_y = x_coresets[0], y_coresets[0]
  for i in range(1, len(x_coresets)):
    merged_x = np.vstack((merged_x, x_coresets[i]))
    merged_y = np.hstack((merged_y, y_coresets[i]))
  return merged_x, merged_y


def get_coreset(x_coresets, y_coresets, single_head, coreset_size = 5000, task_id=0):
  if single_head:
    return merge_coresets(x_coresets, y_coresets)
  else:
    return x_coresets, y_coresets


def get_scores(model, x_testsets, y_testsets, no_epochs, single_head,  x_coresets, y_coresets, batch_size=None, just_vanilla = False):
  acc = []
  if single_head:
    if len(x_coresets) > 0:
      x_train, y_train = get_coreset(x_coresets, y_coresets, single_head, coreset_size = 6000, task_id=0)

      bsize = x_train.shape[0] if (batch_size is None) else batch_size
      x_train = torch.Tensor(x_train)
      y_train = torch.Tensor(y_train)
      model.train(x_train, y_train, 0, no_epochs, bsize, on_coreset = True)

  for i in range(len(x_testsets)):
    if not single_head:
      if len(x_coresets)>0:
        model.load_weights()
        x_train, y_train = get_coreset(x_coresets[i], y_coresets[i], single_head, coreset_size = 6000, task_id=i)
        bsize = x_train.shape[0] if (batch_size is None) else batch_size
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)
        model.train(x_train, y_train, i, no_epochs, bsize, on_coreset = True)

    head = 0 if single_head else i
    x_test, y_test = x_testsets[i], y_testsets[i]
    N = x_test.shape[0]
    bsize = N if (batch_size is None) else batch_size
    cur_acc = 0
    total_batch = int(np.ceil(N * 1.0 / bsize))
    # Loop over all batches
    for i in range(total_batch):
      start_ind = i*bsize
      end_ind = np.min([(i+1)*bsize, N])
      batch_x_test = torch.Tensor(x_test[start_ind:end_ind, :]).to(device = device)
      batch_y_test = torch.Tensor(y_test[start_ind:end_ind]).type(torch.LongTensor).to(device = device)
      pred = model.prediction_prob(batch_x_test, head)
      if not just_vanilla:
        pred_mean = pred.mean(0)
      else:
        pred_mean = pred
      pred_y = torch.argmax(pred_mean, dim=1)
      cur_acc += end_ind - start_ind-(pred_y - batch_y_test).nonzero().shape[0]

    cur_acc = float(cur_acc)
    cur_acc /= N
    acc.append(cur_acc)
    print("Accuracy is {}".format(cur_acc))
  return acc

def concatenate_results(score, all_score):
  if all_score.size == 0:
    all_score = np.reshape(score, (1,-1))
  else:
    new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
    new_arr[:] = np.nan
    new_arr[:,:-1] = all_score
    all_score = np.vstack((new_arr, score))
  return all_score