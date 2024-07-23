import numpy as np
from models import Vanilla_NN, MFVI_NN
from model_utils import get_scores, concatenate_results

### Regularization options = ["js_div", "b_dist", "bd_on_coreset", "kl_div"] ###
### Defaults to kl_div ###
def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, learning_rate=0.001, regularization = 'kl_div', device='cpu'):
  in_dim, out_dim = data_gen.get_dims()
  x_coresets, y_coresets = [], []
  x_testsets, y_testsets = [], []

  all_acc = np.array([])

  task_0 = True

  for task_id in range(data_gen.max_iter):
    x_train, y_train, x_test, y_test = data_gen.next_task()
    x_testsets.append(x_test)
    y_testsets.append(y_test)

    # Set the readout head to train
    head = 0 if single_head else task_id
    bsize = x_train.shape[0] if (batch_size is None) else batch_size

    # Train network with maximum likelihood to initialize first model
    if task_id == 0:
      print_graph_bol = False #set to True if you want to see the graph
      ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0], device=device)
      ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
      mf_weights = ml_model.get_weights()
      mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head=single_head, prev_means=mf_weights, learning_rate=learning_rate, regularization=regularization, device=device)

    if coreset_size > 0:
      x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)

    mf_model.train(x_train, y_train, head, no_epochs, bsize)

    mf_model.update_prior()

    # Save weights before test (and last-minute training on coreset)
    mf_model.save_weights()

    acc = get_scores(mf_model, x_testsets, y_testsets, no_epochs, single_head, x_coresets, y_coresets, batch_size, False)
    all_acc = concatenate_results(acc, all_acc)

    mf_model.load_weights()
    mf_model.clean_copy_weights()


    if not single_head:
      mf_model.create_head()

  return all_acc

def run_coreset_only(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, device='cpu'):
  in_dim, out_dim = data_gen.get_dims()
  x_coresets, y_coresets = [], []
  x_testsets, y_testsets = [], []
  all_acc = np.array([])

  for task_id in range(data_gen.max_iter):
    x_train, y_train, x_test, y_test = data_gen.next_task()
    x_testsets.append(x_test)
    y_testsets.append(y_test)

    head = 0 if single_head else task_id
    bsize = x_train.shape[0] if (batch_size is None) else batch_size

    if task_id == 0:
      mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head = single_head, prev_means=None, device=device)

    if coreset_size > 0:
      x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)


    mf_model.save_weights()

    acc = get_scores(mf_model, x_testsets, y_testsets, no_epochs, single_head, x_coresets, y_coresets, batch_size, just_vanilla =False)

    all_acc = concatenate_results(acc, all_acc)

    mf_model.load_weights()
    mf_model.clean_copy_weights()

    if not single_head:
      mf_model.create_head()

  return all_acc

