import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from model_utils import truncated_normal, init_tensor


class Cla_NN(object):
  def __init__(self, input_size, hidden_size, output_size, training_size, device='cpu'):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.training_size = training_size
    self.device = device
    return


  def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=5, on_coreset = False):
    N = x_train.shape[0]
    self.training_size = N
    if batch_size > N:
        batch_size = N

    costs = []
    # Training cycle
    for epoch in range(no_epochs):
      perm_inds = np.arange(x_train.shape[0])
      np.random.shuffle(perm_inds)
      cur_x_train = x_train[perm_inds]
      cur_y_train = y_train[perm_inds]

      avg_cost = 0.
      total_batch = int(np.ceil(N * 1.0 / batch_size))
      # Loop over all batches
      for i in range(total_batch):
        start_ind = i*batch_size
        end_ind = np.min([(i+1)*batch_size, N])
        batch_x = torch.Tensor(cur_x_train[start_ind:end_ind, :]).to(device = self.device)
        batch_y = torch.Tensor(cur_y_train[start_ind:end_ind]).to(device = self.device)

        self.optimizer.zero_grad()
        cost = self.get_loss(batch_x, batch_y, task_idx, on_coreset = on_coreset)
        cost.backward()
        self.optimizer.step()

        # Compute average loss
        cost = cost.detach()
        avg_cost += cost / total_batch
      # Display logs per epoch step
      if epoch % display_epoch == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost))
      costs.append(avg_cost)

    print("Optimization Finished!")
    return costs

  def prediction_prob(self, x_test, task_idx):
    prob = F.softmax(self._prediction(x_test, task_idx, self.no_pred_samples), dim=-1)
    return prob

''' Neural Network Model '''
class Vanilla_NN(Cla_NN):
  def __init__(self, input_size, hidden_size, output_size, training_size, learning_rate=0.001, device='cpu'):
    super(Vanilla_NN, self).__init__(input_size, hidden_size, output_size, training_size)
    ## init weights and biases
    self.W, self.b, self.W_last, self.b_last, self.size = self.create_weights(
              input_size, hidden_size, output_size)
    self.no_layers = len(hidden_size) + 1
    self.weights = self.W + self.b + self.W_last + self.b_last
    self.training_size = training_size
    self.optimizer = optim.Adam(self.weights, lr=learning_rate)
    self.device = device

  def _prediction(self, inputs, task_idx):
    act = inputs
    for i in range(self.no_layers-1):
        pre = torch.add(torch.matmul(act, self.W[i]), self.b[i])
        act = F.relu(pre)
    pre = torch.add(torch.matmul(act, self.W_last[task_idx]), self.b_last[task_idx])
    return pre

  def _logpred(self, inputs, targets, task_idx):
    loss = torch.nn.CrossEntropyLoss()
    pred = self._prediction(inputs, task_idx)
    log_lik = - loss(pred, targets.type(torch.long))
    return log_lik

  def prediction_prob(self, x_test, task_idx):
    prob = F.softmax(self._prediction(x_test, task_idx), dim=-1)
    return prob

  def get_loss(self, batch_x, batch_y, task_idx, on_coreset = False):
    return -self._logpred(batch_x, batch_y, task_idx)

  def create_weights(self, in_dim, hidden_size, out_dim):
    hidden_size = deepcopy(hidden_size)
    hidden_size.append(out_dim)
    hidden_size.insert(0, in_dim)

    no_layers = len(hidden_size) - 1
    W = []
    b = []
    W_last = []
    b_last = []
    for i in range(no_layers-1):
      din = hidden_size[i]
      dout = hidden_size[i+1]

      #Initializiation values of means
      Wi_m = truncated_normal([din, dout], stddev=0.1, variable = True)
      bi_m = truncated_normal([dout], stddev=0.1, variable = True)

      #Append to list weights
      W.append(Wi_m)
      b.append(bi_m)

    Wi = truncated_normal([hidden_size[-2], out_dim], stddev=0.1, variable = True)
    bi = truncated_normal([out_dim], stddev=0.1, variable = True)
    W_last.append(Wi)
    b_last.append(bi)
    return W, b, W_last, b_last, hidden_size

  def get_weights(self):
    weights = [self.weights[:self.no_layers-1], self.weights[self.no_layers-1:2*(self.no_layers-1)], [self.weights[-2]], [self.weights[-1]]]
    return weights

''' Bayesian Neural Network with Mean field VI approximation '''
class MFVI_NN(Cla_NN):
  def __init__(self, input_size, hidden_size, output_size, training_size,
    no_train_samples=10, no_pred_samples=100, single_head = False, prev_means=None, learning_rate=0.001, regularization = 'kl_div', device='cpu'):
    super(MFVI_NN, self).__init__(input_size, hidden_size, output_size, training_size)

    m1, v1, hidden_size = self.create_weights(input_size, hidden_size, output_size, prev_means)

    self.device = device

    self.input_size = input_size
    self.out_size = output_size
    self.size = hidden_size
    self.single_head = single_head
    self.regularization = regularization

    self.W_m, self.b_m = m1[0], m1[1]
    self.W_v, self.b_v = v1[0], v1[1]

    self.W_last_m, self.b_last_m = [], []
    self.W_last_v, self.b_last_v = [], []

    m2, v2 = self.create_prior(input_size, self.size, output_size)

    self.prior_W_m, self.prior_b_m, = m2[0], m2[1]
    self.prior_W_v, self.prior_b_v = v2[0], v2[1]

    self.prior_W_last_m, self.prior_b_last_m = [], []
    self.prior_W_last_v, self.prior_b_last_v = [], []

    self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
    self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
    self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
    self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None

    self.no_layers = len(self.size) - 1
    self.no_train_samples = no_train_samples
    self.no_pred_samples = no_pred_samples
    self.training_size = training_size
    self.learning_rate = learning_rate

    if prev_means is not None:
        self.init_first_head(prev_means)
    else:
        self.create_head()

    m1.append(self.W_last_m)
    m1.append(self.b_last_m)
    v1.append(self.W_last_v)
    v1.append(self.b_last_v)

    r1 = m1 + v1
    self.weights = [item for sublist in r1 for item in sublist]

    self.optimizer = optim.Adam(self.weights, lr=learning_rate)

  # projection term for model selected here
  def get_loss(self, batch_x, batch_y, task_idx, on_coreset = False):
    if self.regularization == 'js_div':
      return torch.div(self._JSD_term(), self.training_size) - self._logpred(batch_x, batch_y, task_idx)
    elif self.regularization == 'b_dist':
      return torch.div(self._BD_term(), self.training_size) - self._logpred(batch_x, batch_y, task_idx)
    elif self.regularization == 'bd_on_corset' and on_coreset:
      return torch.div(self._BD_term(), self.training_size) - self._logpred(batch_x, batch_y, task_idx)
    else:
      return torch.div(self._KL_term(), self.training_size) - self._logpred(batch_x, batch_y, task_idx)

  def _prediction(self, inputs, task_idx, no_samples):
    K = no_samples
    size = self.size

    act = torch.unsqueeze(inputs, 0).repeat([K, 1, 1])
    for i in range(self.no_layers-1):
      din = self.size[i]
      dout = self.size[i+1]
      eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = self.device)
      eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = self.device)
      weights = torch.add(eps_w * torch.exp(0.5*self.W_v[i]), self.W_m[i])
      biases = torch.add(eps_b * torch.exp(0.5*self.b_v[i]), self.b_m[i])
      pre = torch.add(torch.einsum('mni,mio->mno', act, weights), biases)
      act = F.relu(pre)

    din = self.size[-2]
    dout = self.size[-1]

    eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = self.device)
    eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = self.device)
    Wtask_m = self.W_last_m[task_idx]
    Wtask_v = self.W_last_v[task_idx]
    btask_m = self.b_last_m[task_idx]
    btask_v = self.b_last_v[task_idx]

    weights = torch.add(eps_w * torch.exp(0.5*Wtask_v),Wtask_m)
    biases = torch.add(eps_b * torch.exp(0.5*btask_v), btask_m)
    act = torch.unsqueeze(act, 3)
    weights = torch.unsqueeze(weights, 1)
    pre = torch.add(torch.sum(act * weights, dim = 2), biases)
    return pre

  def _logpred(self, inputs, targets, task_idx):
    loss = torch.nn.CrossEntropyLoss()
    pred = self._prediction(inputs, task_idx, self.no_train_samples).view(-1,self.out_size)
    targets = targets.repeat([self.no_train_samples, 1]).view(-1)
    log_liks = -loss(pred, targets.type(torch.long))
    log_lik = log_liks.mean()
    return log_lik

  ''' Bhattacharyya distance '''
  def _BD_term(self):
    # compute BD iteratively by layer
    bd = 0
    for i in range(self.no_layers-1):
      din = self.size[i]
      dout = self.size[i+1]
      const_term = dout * din

      m_q, log_v_q = self.W_m[i], self.W_v[i] # Mean and log var for current layer's weights
      m_p, v_p = self.prior_W_m[i], self.prior_W_v[i]

      # Convert log var to actual var for posterior
      v_q = torch.exp(log_v_q)

      # Bhattacharyya distance for weights
      mean_diff = m_p - m_q
      sigma_sum = v_q + v_p

      bd_weights = 0.125 * torch.sum((mean_diff ** 2) / sigma_sum) + 0.5 * torch.sum(torch.log(sigma_sum / (2 * torch.sqrt(v_q * v_p))))

      # Update total Bhattacharyya distance
      bd += const_term + bd_weights

      # Repeat for biases
      m_q, log_v_q = self.b_m[i], self.b_v[i]
      m_p, v_p = self.prior_b_m[i], self.prior_b_v[i]
      v_q = torch.exp(log_v_q)

      mean_diff = m_p - m_q
      sigma_sum = v_q + v_p
      bd_biases = 0.125 * torch.sum((mean_diff ** 2) / sigma_sum) + 0.5 * torch.sum(torch.log(sigma_sum / (2 * torch.sqrt(v_q * v_p))))

      # Update total Bhattacharyya distance
      bd += const_term + bd_biases

    no_tasks = len(self.W_last_m)
    din = self.size[-2]
    dout = self.size[-1]
    const_term = dout * din

    # for number of tasks:
    for i in range(no_tasks):
      # For weights of the last layer
      m_q, log_v_q = self.W_last_m[i], self.W_last_v[i]
      m_p, v_p = self.prior_W_last_m[i], self.prior_W_last_v[i]

      # Convert log var to actual var for posterior
      v_q = torch.exp(log_v_q)

      # Bhattacharyya distance for weights
      mean_diff = m_p - m_q
      sigma_sum = v_q + v_p
      bd_weights = 0.125 * torch.sum((mean_diff ** 2) / sigma_sum) + 0.5 * torch.sum(torch.log(sigma_sum / (2 * torch.sqrt(v_q * v_p))))

      # Update total Bhattacharyya distance
      bd += const_term + bd_weights

      # Repeat for biases
      m_q, log_v_q = self.b_last_m[i], self.b_last_v[i]
      m_p, v_p = self.prior_b_last_m[i], self.prior_b_last_v[i]
      v_q = torch.exp(log_v_q)

      mean_diff = m_p - m_q
      sigma_sum = v_q + v_p
      bd_biases = 0.125 * torch.sum((mean_diff ** 2) / sigma_sum) + 0.5 * torch.sum(torch.log(sigma_sum / (2 * torch.sqrt(v_q * v_p))))

      # Update total Bhattacharyya distance
      bd += const_term + bd_biases

    return bd

  ''' Jensen - Shannon divergence'''
  def _JSD_term(self):
    def _KL_div_helper(mu1, log_var1, mu2, log_var2):
      # Compute the KL divergence between two Gaussian distributions
      return 0.5 * torch.sum(torch.exp(log_var1 - log_var2) + (mu2 - mu1)**2 / torch.exp(log_var2) - 1 + (log_var2 - log_var1))

    # compute JSD iteratively by layer
    jsd = 0
    for i in range(self.no_layers-1):
      # Extract means and log variances for weights and biases from both distributions
      m_p, log_v_p = self.prior_W_m[i], torch.log(self.prior_W_v[i])  # Prior mean and log var
      m_q, log_v_q = self.W_m[i], self.W_v[i] # Posterior mean and log var
      din = self.size[i]
      dout = self.size[i+1]
      const_term = dout * din

      # Compute the mixture distribution parameters for weights
      m_m = 0.5 * (m_p + m_q)
      log_v_m = torch.log(0.5 * (torch.exp(log_v_p) + torch.exp(log_v_q)))  # Log var of mixture

      # Compute Jensen-Shannon divergence for weights
      jsd += const_term + 0.5 * (_KL_div_helper(m_p, log_v_p, m_m, log_v_m) + _KL_div_helper(m_q, log_v_q, m_m, log_v_m))

      # Repeat for biases
      m_p, log_v_p = self.prior_b_m[i], torch.log(self.prior_b_v[i])  # Prior mean and log var for biases
      m_q, log_v_q = self.b_m[i], self.b_v[i] # Posterior mean and log var for biases

      # Compute the mixture distribution parameters for biases
      m_m = 0.5 * (m_p + m_q)
      log_v_m = torch.log(0.5 * (torch.exp(log_v_p) + torch.exp(log_v_q)))  # Log var of mixture

      # Compute Jensen-Shannon divergence for biases
      jsd += const_term + 0.5 * (_KL_div_helper(m_p, log_v_p, m_m, log_v_m) + _KL_div_helper(m_q, log_v_q, m_m, log_v_m))

    no_tasks = len(self.W_last_m)
    din = self.size[-2]
    dout = self.size[-1]
    const_term = dout * din

    for i in range(no_tasks):
      m_q, log_v_q = self.W_last_m[i], self.W_last_v[i]
      m_p, log_v_p = self.prior_W_last_m[i], torch.log(self.prior_W_last_v[i])

      m_m = 0.5 * (m_p + m_q)
      log_v_m = torch.log(0.5 * (torch.exp(log_v_p) + torch.exp(log_v_q)))

      jsd += const_term + 0.5 * (_KL_div_helper(m_p, log_v_p, m_m, log_v_m) + _KL_div_helper(m_q, log_v_q, m_m, log_v_m))

      # Repeat for biases
      m_q, log_v_q = self.b_last_m[i], self.b_last_v[i]
      m_p, log_v_p = self.prior_b_last_m[i], torch.log(self.prior_b_last_v[i])

      m_m = 0.5 * (m_p + m_q)
      log_v_m = torch.log(0.5 * (torch.exp(log_v_p) + torch.exp(log_v_q)))

      jsd += const_term + 0.5 * (_KL_div_helper(m_p, log_v_p, m_m, log_v_m) + _KL_div_helper(m_q, log_v_q, m_m, log_v_m))

    return jsd

  ''' KL divergence'''
  def _KL_term(self):
    # compute KL iteratively by layer
    kl = 0
    for i in range(self.no_layers-1):
      din = self.size[i]
      dout = self.size[i+1]
      m_q, log_v_q = self.W_m[i], self.W_v[i] # posterior, v is log variance
      m_p, v_p = self.prior_W_m[i], self.prior_W_v[i]
      const_term = -0.5 * dout * din

      log_std_diff = 0.5 * torch.sum(torch.log(v_p) - log_v_q)
      # torch.exp(v) converts from log var back to var
      mu_diff_term = 0.5 * torch.sum((torch.exp(log_v_q) + (m_p - m_q)**2) / v_p)
      kl += const_term + log_std_diff + mu_diff_term

      m_q, log_v_q = self.b_m[i], self.b_v[i]
      m_p, v_p = self.prior_b_m[i], self.prior_b_v[i]

      const_term = -0.5 * dout
      log_std_diff = 0.5 * torch.sum(torch.log(v_p) - log_v_q)
      mu_diff_term = 0.5 * torch.sum((torch.exp(log_v_q) + (m_p - m_q)**2) / v_p)
      kl +=  log_std_diff + mu_diff_term + const_term

    no_tasks = len(self.W_last_m)
    din = self.size[-2]
    dout = self.size[-1]

    for i in range(no_tasks):
      m_q, log_v_q = self.W_last_m[i], self.W_last_v[i]
      m_p, v_p = self.prior_W_last_m[i], self.prior_W_last_v[i]

      const_term = - 0.5 * dout * din
      log_std_diff = 0.5 * torch.sum(torch.log(v_p) - log_v_q)
      mu_diff_term = 0.5 * torch.sum((torch.exp(log_v_q) + (m_p - m_q)**2) / v_p)
      kl += const_term + log_std_diff + mu_diff_term

      m_q, log_v_q = self.b_last_m[i], self.b_last_v[i]
      m_p, v_p = self.prior_b_last_m[i], self.prior_b_last_v[i]

      const_term = -0.5 * dout
      log_std_diff = 0.5 * torch.sum(torch.log(v_p) - log_v_q)
      mu_diff_term = 0.5 * torch.sum((torch.exp(log_v_q) + (m_p - m_q)**2) / v_p)
      kl += const_term + log_std_diff + mu_diff_term

    return kl

  def save_weights(self):
    ''' Save weights before training on the coreset before getting the test accuracy '''

    print("Saving weights before core set training")
    self.W_m_copy = [self.W_m[i].clone().detach().data for i in range(len(self.W_m))]
    self.W_v_copy = [self.W_v[i].clone().detach().data for i in range(len(self.W_v))]
    self.b_m_copy = [self.b_m[i].clone().detach().data for i in range(len(self.b_m))]
    self.b_v_copy = [self.b_v[i].clone().detach().data for i in range(len(self.b_v))]

    self.W_last_m_copy = [self.W_last_m[i].clone().detach().data for i in range(len(self.W_last_m))]
    self.W_last_v_copy = [self.W_last_v[i].clone().detach().data for i in range(len(self.W_last_v))]
    self.b_last_m_copy = [self.b_last_m[i].clone().detach().data for i in range(len(self.b_last_m))]
    self.b_last_v_copy = [self.b_last_v[i].clone().detach().data for i in range(len(self.b_last_v))]

    self.prior_W_m_copy = [self.prior_W_m[i].data for i in range(len(self.prior_W_m))]
    self.prior_W_v_copy = [self.prior_W_v[i].data for i in range(len(self.prior_W_v))]
    self.prior_b_m_copy = [self.prior_b_m[i].data for i in range(len(self.prior_b_m))]
    self.prior_b_v_copy = [self.prior_b_v[i].data for i in range(len(self.prior_b_v))]

    self.prior_W_last_m_copy = [self.prior_W_last_m[i].data for i in range(len(self.prior_W_last_m))]
    self.prior_W_last_v_copy = [self.prior_W_last_v[i].data for i in range(len(self.prior_W_last_v))]
    self.prior_b_last_m_copy = [self.prior_b_last_m[i].data for i in range(len(self.prior_b_last_m))]
    self.prior_b_last_v_copy = [self.prior_b_last_v[i].data for i in range(len(self.prior_b_last_v))]

    return

  def load_weights(self):
    ''' Re-load weights after getting the test accuracy '''

    print("Reloading previous weights after core set training")
    self.weights = []
    self.W_m = [self.W_m_copy[i].clone().detach().data for i in range(len(self.W_m))]
    self.W_v = [self.W_v_copy[i].clone().detach().data for i in range(len(self.W_v))]
    self.b_m = [self.b_m_copy[i].clone().detach().data for i in range(len(self.b_m))]
    self.b_v = [self.b_v_copy[i].clone().detach().data for i in range(len(self.b_v))]

    for i in range(len(self.W_m)):
      self.W_m[i].requires_grad = True
      self.W_v[i].requires_grad = True
      self.b_m[i].requires_grad = True
      self.b_v[i].requires_grad = True

    self.weights += self.W_m
    self.weights += self.W_v
    self.weights += self.b_m
    self.weights += self.b_v

    self.W_last_m = [self.W_last_m_copy[i].clone().detach().data for i in range(len(self.W_last_m))]
    self.W_last_v = [self.W_last_v_copy[i].clone().detach().data for i in range(len(self.W_last_v))]
    self.b_last_m = [self.b_last_m_copy[i].clone().detach().data for i in range(len(self.b_last_m))]
    self.b_last_v = [self.b_last_v_copy[i].clone().detach().data for i in range(len(self.b_last_v))]

    for i in range(len(self.W_last_m)):
      self.W_last_m[i].requires_grad = True
      self.W_last_v[i].requires_grad = True
      self.b_last_m[i].requires_grad = True
      self.b_last_v[i].requires_grad = True

    self.weights += self.W_last_m
    self.weights += self.W_last_v
    self.weights += self.b_last_m
    self.weights += self.b_last_v

    self.optimizer = optim.Adam(self.weights, lr=self.learning_rate)
    self.prior_W_m = [self.prior_W_m_copy[i].data for i in range(len(self.prior_W_m))]
    self.prior_W_v = [self.prior_W_v_copy[i].data for i in range(len(self.prior_W_v))]
    self.prior_b_m = [self.prior_b_m_copy[i].data for i in range(len(self.prior_b_m))]
    self.prior_b_v = [self.prior_b_v_copy[i].data for i in range(len(self.prior_b_v))]

    self.prior_W_last_m = [self.prior_W_last_m_copy[i].data for i in range(len(self.prior_W_last_m))]
    self.prior_W_last_v = [self.prior_W_last_v_copy[i].data for i in range(len(self.prior_W_last_v))]
    self.prior_b_last_m = [self.prior_b_last_m_copy[i].data for i in range(len(self.prior_b_last_m))]
    self.prior_b_last_v = [self.prior_b_last_v_copy[i].data for i in range(len(self.prior_b_last_v))]

    return

  def clean_copy_weights(self):
    self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
    self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
    self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
    self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None

  def create_head(self):
    # Create new head when a new task is detected
    print("creating a new head")
    din = self.size[-2]
    dout = self.size[-1]

    W_m= truncated_normal([din, dout], stddev=0.1, variable=True)
    b_m= truncated_normal([dout], stddev=0.1, variable=True)
    W_v = init_tensor(-6.0,  dout = dout, din = din, variable= True)
    b_v = init_tensor(-6.0,  dout = dout, variable= True)

    self.W_last_m.append(W_m)
    self.W_last_v.append(W_v)
    self.b_last_m.append(b_m)
    self.b_last_v.append(b_v)


    W_m_p = torch.zeros([din, dout]).to(device = self.device)
    b_m_p = torch.zeros([dout]).to(device = self.device)
    W_v_p =  init_tensor(1,  dout = dout, din = din)
    b_v_p = init_tensor(1, dout = dout)

    self.prior_W_last_m.append(W_m_p)
    self.prior_W_last_v.append(W_v_p)
    self.prior_b_last_m.append(b_m_p)
    self.prior_b_last_v.append(b_v_p)
    self.weights = []
    self.weights += self.W_m
    self.weights += self.W_v
    self.weights += self.b_m
    self.weights += self.b_v
    self.weights += self.W_last_m
    self.weights += self.W_last_v
    self.weights += self.b_last_m
    self.weights += self.b_last_v
    self.optimizer = optim.Adam(self.weights, lr=self.learning_rate)

    return


  def init_first_head(self, prev_means):
    # When the MFVI_NN is instanciated, we initialize weights with those of the Vanilla NN
    print("initializing first head")
    din = self.size[-2]
    dout = self.size[-1]
    self.prior_W_last_m = [torch.zeros([din, dout]).to(device = self.device)]
    self.prior_b_last_m = [torch.zeros([dout]).to(device = self.device)]
    self.prior_W_last_v =  [init_tensor(1,  dout = dout, din = din)]
    self.prior_b_last_v = [init_tensor(1, dout = dout)]

    W_last_m = prev_means[2][0].detach().data
    W_last_m.requires_grad = True
    self.W_last_m = [W_last_m]
    self.W_last_v = [init_tensor(-6.0,  dout = dout, din = din, variable= True)]


    b_last_m = prev_means[3][0].detach().data
    b_last_m.requires_grad = True
    self.b_last_m = [b_last_m]
    self.b_last_v = [init_tensor(-6.0, dout = dout, variable= True)]

    return

  def create_weights(self, in_dim, hidden_size, out_dim, prev_means):
    hidden_size = deepcopy(hidden_size)
    hidden_size.append(out_dim)
    hidden_size.insert(0, in_dim)

    no_layers = len(hidden_size) - 1
    W_m = []
    b_m = []
    W_v = []
    b_v = []

    for i in range(no_layers-1):
      din = hidden_size[i]
      dout = hidden_size[i+1]
      if prev_means is not None:
          W_m_i = prev_means[0][i].detach().data
          W_m_i.requires_grad = True
          bi_m_i = prev_means[1][i].detach().data
          bi_m_i.requires_grad = True
      else:
      #Initializiation values of means
          W_m_i= truncated_normal([din, dout], stddev=0.1, variable=True)
          bi_m_i= truncated_normal([dout], stddev=0.1, variable=True)
      #Initializiation values of variances
      W_v_i = init_tensor(-6.0,  dout = dout, din = din, variable = True)
      bi_v_i = init_tensor(-6.0,  dout = dout, variable = True)

      #Append to list weights
      W_m.append(W_m_i)
      b_m.append(bi_m_i)
      W_v.append(W_v_i)
      b_v.append(bi_v_i)

    return [W_m, b_m], [W_v, b_v], hidden_size

  def create_prior(self, in_dim, hidden_size, out_dim, initial_mean = 0, initial_variance = 1):
    no_layers = len(hidden_size) - 1
    W_m = []
    b_m = []

    W_v = []
    b_v = []

    for i in range(no_layers - 1):
        din = hidden_size[i]
        dout = hidden_size[i + 1]

        # Initializiation values of means
        W_m_val = initial_mean * torch.zeros([din, dout]).to(device = self.device)
        bi_m_val = initial_mean * torch.zeros([dout]).to(device = self.device)

        # Initializiation values of variances
        W_v_val = initial_variance * init_tensor(1,  dout = dout, din = din )
        bi_v_val =  initial_variance * init_tensor(1,  dout = dout)

        # Append to list weights
        W_m.append(W_m_val)
        b_m.append(bi_m_val)
        W_v.append(W_v_val)
        b_v.append(bi_v_val)

    return [W_m, b_m], [W_v, b_v]


  def update_prior(self):
    print("updating prior...")
    for i in range(len(self.W_m)):
      self.prior_W_m[i].data.copy_(self.W_m[i].clone().detach().data)
      self.prior_b_m[i].data.copy_(self.b_m[i].clone().detach().data)
      self.prior_W_v[i].data.copy_(torch.exp(self.W_v[i].clone().detach().data))
      self.prior_b_v[i].data.copy_(torch.exp(self.b_v[i].clone().detach().data))

    length = len(self.W_last_m)

    for i in range(length):
      self.prior_W_last_m[i].data.copy_(self.W_last_m[i].clone().detach().data)
      self.prior_b_last_m[i].data.copy_(self.b_last_m[i].clone().detach().data)
      self.prior_W_last_v[i].data.copy_(torch.exp(self.W_last_v[i].clone().detach().data))
      self.prior_b_last_v[i].data.copy_(torch.exp(self.b_last_v[i].clone().detach().data))

    return
     