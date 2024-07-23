import os
import torch
import numpy as np
from dataset_generator import PermutedMnistGenerator, SplitMnistGenerator
from model_utils import rand_from_batch, k_center
from train import run_vcl, run_coreset_only

# Create a directory
dir_path = "./results"
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

# Find device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    ''' Run Permuted '''
    np.random.seed(1)
    hidden_size = [100, 100]
    batch_size = 256
    no_epochs = 100
    single_head = True
    num_tasks = 10

    #Just VCL
    coreset_size = 0
    data_gen = PermutedMnistGenerator(num_tasks)
    vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'kl_div', device=device)
    np.save("./results/paper_results/permutedMNIST/VCL{}".format(""), vcl_result)
    print(vcl_result)

    coreset_size = 0
    data_gen = PermutedMnistGenerator(num_tasks)
    vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'b_dist', device=device)
    np.save("./results/experimental_results/permutedMNIST/BD_VCL{}".format(""), vcl_result)
    print(vcl_result)

    coreset_size = 0
    data_gen = PermutedMnistGenerator(num_tasks)
    vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'js_div', device=device)
    np.save("./results/experimental_results/permutedMNIST/JSD_VCL{}".format(""), vcl_result)
    print(vcl_result)

    #VCL + Random Coreset'
    coreset_size = 1000
    data_gen = PermutedMnistGenerator(num_tasks)
    rand_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'kl_div', device=device)
    np.save("./results/paper_results/permutedMNIST/rand-VCL-{}".format(coreset_size), rand_vcl_result)
    print(rand_vcl_result)

    coreset_size = 1000
    data_gen = PermutedMnistGenerator(num_tasks)
    rand_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'b_dist', device=device)
    np.save("./results/experimental_results/permutedMNIST/rand-BD-VCL-{}".format(coreset_size), rand_vcl_result)
    print(rand_vcl_result)

    coreset_size = 1000
    data_gen = PermutedMnistGenerator(num_tasks)
    rand_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'js_div', device=device)
    np.save("./results/experimental_results/permutedMNIST/rand-JSD-VCL-{}".format(coreset_size), rand_vcl_result)
    print(rand_vcl_result)

    coreset_size = 1000
    data_gen = PermutedMnistGenerator(num_tasks)
    rand_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'bd_on_coreset', device=device)
    np.save("./results/experimental_results/permutedMNIST/rand-coresetBD-VCL-{}".format(coreset_size), rand_vcl_result)
    print(rand_vcl_result)

    #VCL + k-center coreset
    coreset_size = 200
    data_gen = PermutedMnistGenerator(num_tasks)
    kcen_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, k_center, coreset_size, batch_size, single_head, regularization = 'kl_div', device=device)
    print(kcen_vcl_result)
    np.save("./results/paper_results/permutedMNIST/kcen-VCL{}".format(coreset_size), kcen_vcl_result)

    coreset_size = 200
    data_gen = PermutedMnistGenerator(num_tasks)
    kcen_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, k_center, coreset_size, batch_size, single_head, regularization = 'bd_on_coreset', device=device)
    print(kcen_vcl_result)
    np.save("./results/experimental_results/permutedMNIST/kcen-coresetBD-VCL{}".format(coreset_size), kcen_vcl_result)

    ''' Run Permuted Just Coreset '''
    np.random.seed(0)
    hidden_size = [100, 100]
    batch_size = 256
    no_epochs = 100
    single_head = True
    num_tasks = 10

    coreset_size = 200
    data_gen = PermutedMnistGenerator(num_tasks)
    kcen_vcl_result = run_coreset_only(hidden_size, no_epochs, data_gen, k_center, coreset_size, batch_size, single_head, device=device)
    print(kcen_vcl_result)
    np.save("./results/paper_results/permutedMNIST/kcen-coreset-only{}".format(coreset_size), kcen_vcl_result)

    ''' Run Split '''
    np.random.seed(0)
    hidden_size = [256, 256]
    batch_size = None
    no_epochs = 120
    single_head = False
    run_coreset_only = False

    #Just VCL
    coreset_size = 0
    data_gen = SplitMnistGenerator()
    vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'bd_on_coreset', device=device)
    np.save("./results/experimental_results/split-MNIST/coresetBD-VCL-split{}".format(""), vcl_result)

    coreset_size = 0
    data_gen = SplitMnistGenerator()
    vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'kl_div', device=device)
    np.save("./results/paper_results/split-MNIST/VCL-split{}".format(""), vcl_result)

    coreset_size = 0
    data_gen = SplitMnistGenerator()
    vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'js_div', device=device)
    np.save("./results/experimental_results/split-MNIST/JSD-VCL-split{}".format(""), vcl_result)

    coreset_size = 0
    data_gen = SplitMnistGenerator()
    vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'b_dist', device=device)
    np.save("./results/experimental_results/split-MNIST/BD-VCL-split{}".format(""), vcl_result)

    #VCL + Random Coreset
    coreset_size = 40
    data_gen = SplitMnistGenerator()
    rand_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'kl_div', device=device)
    print(rand_vcl_result)
    np.save("./results/paper_results/split-MNIST/rand-VCL-split{}".format(""), rand_vcl_result)

    coreset_size = 40
    data_gen = SplitMnistGenerator()
    rand_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'b_dist', device=device)
    print(rand_vcl_result)
    np.save("./results/experimental_results/split-MNIST/rand-BD-VCL-split{}".format(""), rand_vcl_result)

    coreset_size = 40
    data_gen = SplitMnistGenerator()
    rand_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'js_div', device=device)
    print(rand_vcl_result)
    np.save("./results/experimental_results/split-MNIST/rand-JSD-VCL-split{}".format(""), rand_vcl_result)

    coreset_size = 40
    data_gen = SplitMnistGenerator()
    rand_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, regularization = 'bd_on_coreset', device=device)
    print(rand_vcl_result)
    np.save("./results/experimental_results/split-MNIST/rand-coresetBD-VCL-split{}".format(""), rand_vcl_result)

    #VCL + k-center coreset
    coreset_size = 40
    data_gen = SplitMnistGenerator()
    kcen_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, k_center, coreset_size, batch_size, single_head, regularization = "kl_div", device=device)
    print(kcen_vcl_result)
    np.save("./results/paper_results/split-MNIST/kcenVCL-split{}".format(""), kcen_vcl_result)

    coreset_size = 40
    data_gen = SplitMnistGenerator()
    kcen_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, k_center, coreset_size, batch_size, single_head, regularization = "bd_on_coreset", device=device)
    print(kcen_vcl_result)
    np.save("./results/experimental_results/split-MNIST/kcen-coresetBD-VCL-split{}".format(""), kcen_vcl_result)

    coreset_size = 40
    data_gen = SplitMnistGenerator()
    kcen_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, k_center, coreset_size, batch_size, single_head, regularization = "b_dist", device=device)
    print(kcen_vcl_result)
    np.save("./results/experimental_results/split-MNIST/kcen-BD-VCL-split{}".format(""), kcen_vcl_result)

    coreset_size = 40
    data_gen = SplitMnistGenerator()
    kcen_vcl_result = run_vcl(hidden_size, no_epochs, data_gen, k_center, coreset_size, batch_size, single_head, regularization = "js_div", device=device)
    print(kcen_vcl_result)
    np.save("./results/experimental_results/split-MNIST/kcen-JSD-VCL-split{}".format(""), kcen_vcl_result)

    ''' Run Split Just Coreset '''
    np.random.seed(0)
    hidden_size = [256, 256]
    batch_size = None
    no_epochs = 120
    single_head = False
    coreset_size = 40

    #Random coreset
    data_gen = SplitMnistGenerator()
    rand_vcl_result = run_coreset_only(hidden_size, no_epochs, data_gen, rand_from_batch, coreset_size, batch_size, single_head, device=device)
    print(rand_vcl_result)
    np.save("./results/paper_results/split-MNIST/rand-coreset-only-split{}".format(""), rand_vcl_result)

    #K-center coreset
    data_gen = SplitMnistGenerator()
    kcen_vcl_result = run_coreset_only(hidden_size, no_epochs, data_gen, k_center, coreset_size, batch_size, single_head, device=device)
    print(kcen_vcl_result)
    np.save("./results/paper_results/split-MNIST/kcen-coreset-only-split{}".format(""), kcen_vcl_result)





