import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from PIL import Image
import gzip
import pickle as cp
import glob

directory = "./results/experimental_results/permutedMNIST/"
directory_files = sorted(glob.glob(directory))
BD_VCL= np.load(directory+"BD_VCL.npy")
JSD_VCL = np.load(directory+"JSD_VCL.npy")
coresetBD_VCL = np.load(directory+"rand-coresetBD-VCL-200.npy")
rand_BD_VCL = np.load(directory+"rand-BD-VCL-200.npy")
rand_JSD_VCL = np.load(directory+"rand-JSD-VCL-200.npy")
rand_coresetBD_VCL = np.load(directory+"rand-coresetBD-VCL-200.npy")
kcen_coresetBD_VCL = np.load(directory+"kcen-coresetBD-VCL200.npy")

directory = "./results/paper_results/permutedMNIST/"
VCL = np.load(directory+'VCL.npy')
kcen_VCL = np.load(directory+"kcen-VCL200.npy")
kcen_coreset_only = np.load(directory+"kcen-coreset-only200.npy")
rand_VCL = np.load(directory+"rand-VCL-200.npy")

directory = "./results/paper_results/split-MNIST/"
directory_files = sorted(glob.glob(directory + "*-split.npy"))
VCL_split = np.load(directory+"VCL-split.npy")
randVCL_split = np.load(directory+"rand-VCL-split.npy")
kcenVCL_split = np.load(directory+"kcenVCL-split.npy")
kcen_coreset_only_split = np.load(directory+"kcen-coreset-only-split.npy")
rand_coreset_only_split =  np.load(directory+"rand-coreset-only-split.npy")


directory = "./results/experimental_results/split-MNIST/"
directory_files = sorted(glob.glob(directory + "*-split.npy"))
BD_VCL_split = np.load(directory+"BD-VCL-split.npy")
JSD_VCL_split = np.load(directory+"JSD-VCL-split.npy")
coresetBD_VCL_split = np.load(directory+"coresetBD-VCL-split.npy")
rand_BD_VCL_split = np.load(directory+"rand-BD-VCL-split.npy")
rand_JSD_VCL_split = np.load(directory+"rand-JSD-VCL-split.npy")
rand_coresetBD_VCL_split = np.load(directory+"rand-coresetBD-VCL-split.npy")
kcen_BD_VCL_split = np.load(directory+"kcen-BD-VCL-split.npy")
kcen_JSD_VCL_split = np.load(directory+"kcen-JSD-VCL-split.npy")
kcen_coresetBD_VCL_split = np.load(directory+"kcen-coresetBD-VCL-split.npy")

## Permuted MNIST ##
## paper reported - avg accuracy after 10 tasks ##
paper_VCL = 0.90
paper_EWC = 0.84
paper_SI = 0.86
paper_LP = 0.82
paper_rand_VCL = 0.93
paper_kcen_VCL = 0.93
paper_reported = {"paper_VCL": paper_VCL, "paper_EWC": paper_EWC, "paper_SI": paper_SI, "paper_LP": paper_LP, "paper_rand_VCL": paper_rand_VCL, "paper_kcen_VCL": paper_kcen_VCL}
## paper implementation - avg accuracy after 10 tasks ##
paper_implementation = {"VCL": VCL, "kcen_VCL": kcen_VCL, "kcen_coreset_only": kcen_coreset_only, "rand_VCL": rand_VCL}
## experimental - avg accuracy after 10 tasks ##
experimental = {"BD_VCL": BD_VCL, "JSD_VCL": JSD_VCL, "rand_BD_VCL": rand_BD_VCL, "rand_JSD_VCL": rand_JSD_VCL, "rand_coresetBD_VCL": rand_coresetBD_VCL, "kcen_coresetBD_VCL": kcen_coresetBD_VCL}

## Split MNIST ##
## paper reported - avg accuracy after 5 tasks ##
paper_VCL_split = 0.97
paper_EWC_split = 0.631
paper_SI_split = 0.989
paper_LP_split = 0.612
paper_kcen_VCL_split = 0.984
paper_reported_split = {"paper_VCL_split": paper_VCL_split, "paper_EWC_split": paper_EWC_split, "paper_SI_split": paper_SI_split, "paper_LP_split": paper_LP_split, "paper_kcen_VCL_split": paper_kcen_VCL_split}
## paper implementation - avg accuracy after 5 tasks ##
paper_implementation_split = {"VCL_split": VCL_split, "kcen_coreset_only_split": kcen_coreset_only_split, "kcenVCL_split": kcenVCL_split, "randVCL_split": randVCL_split, "rand_coreset_only_split": rand_coreset_only_split}
## experimental - avg accuracy after 5 tasks ##
experimental_split = {"BD_VCL_split": BD_VCL_split, "JSD_VCL_split": JSD_VCL_split, "coresetBD_VCL_split": coresetBD_VCL_split, "kcen_BD_VCL_split": kcen_BD_VCL_split, "kcen_JSD_VCL_split": kcen_JSD_VCL_split, "kcen_coresetBD_VCL_split": kcen_coresetBD_VCL_split, "rand_BD_VCL_split": rand_BD_VCL_split, "rand_JSD_VCL_split": rand_JSD_VCL_split, "rand_coresetBD_VCL_split": rand_coresetBD_VCL_split}

if __name__ == "__main__":
    print("---- PERMUTED MNIST ----")
    for name, acc in paper_reported.items():
        print(name + ": ", acc)
    for name, acc in paper_implementation.items():
        print(name+": ", np.nanmean(acc,1)[-1])
    for name, acc in experimental.items():
        print(name+": ", np.nanmean(acc,1)[-1])
    print()
    print("---- SPLIT MNIST ----")
    for name, acc in paper_reported_split.items():
        print(name + ": ", acc)
    for name, acc in paper_implementation_split.items():
        print(name+": ", np.nanmean(acc,1)[-1])
    for name, acc in experimental_split.items():
        print(name+": ", np.nanmean(acc,1)[-1])

    ''' Permuted MNIST '''
    ## Average test set accuracy for all original paper methods in the Permuted MNIST experiment ##
    fig = plt.figure(figsize=(20,10))
    ax = plt.gca()
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['lines.linewidth'] = 1.75

    coreset_size = 200

    label = "VCL + Random (200)"
    acc_mean_file = np.nanmean(rand_VCL, 1)
    plt.plot(np.arange(len(acc_mean_file))+1, acc_mean_file, label=label, color ="royalblue" ,linestyle = "--", marker='o')


    label_coreset = "Coreset (200)"
    acc_mean_coreset_file = np.nanmean(kcen_coreset_only, 1)
    plt.plot(np.arange(len(acc_mean_coreset_file))+1, acc_mean_coreset_file, label = label_coreset, color="k", linestyle = "--", marker='d')


    acc_mean = np.nanmean(VCL, 1)
    plt.plot(np.arange(len(acc_mean))+1, acc_mean, label="VCL", color = "b", marker='s')


    acc_mean = np.nanmean(kcen_VCL, 1)
    plt.plot(np.arange(len(acc_mean))+1, acc_mean, label="VCL + k-Center (200)", color = "deepskyblue", linestyle = "--",marker='d')

    ax.set_xticks(range(1, len(acc_mean_file)+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('# tasks')
    ax.legend()
    plt.title("Average test set accuracy for all original paper methods in the Permuted MNIST experiment")
    plt.savefig("paper_replication_results.jpg")
    
    ## Average test set accuracy on Permuted MNIST - experimental comparison ##
    fig = plt.figure(figsize=(20,10))
    ax = plt.gca()
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['lines.linewidth'] = 1.75

    acc_mean = np.nanmean(VCL, 1)
    plt.plot(np.arange(len(acc_mean))+1, acc_mean, label="VCL", color = "b", marker='s')

    acc_mean = np.nanmean(JSD_VCL, 1)
    plt.plot(np.arange(len(acc_mean))+1, acc_mean, label="JSD-VCL", color = "m", marker='s')

    acc_mean = np.nanmean(BD_VCL, 1)
    plt.plot(np.arange(len(acc_mean))+1, acc_mean, label="BD-VCL", color = "g", marker='s')

    ax.set_xticks(range(1, len(acc_mean)+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('# tasks')
    ax.legend()
    plt.title("Average test set accuracy on Permuted MNIST - experimental comparison")
    plt.savefig("regularization_metric_results.jpg")
    
    ## Average test set accuracy on Permuted MNIST - experimental results + coreset ##
    fig = plt.figure(figsize=(20,10))
    ax = plt.gca()
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['lines.linewidth'] = 1.75

    label = "VCL + Random (200)"
    acc_mean_file = np.nanmean(rand_VCL, 1)
    plt.plot(np.arange(len(acc_mean_file))+1, acc_mean_file, label=label, color ="royalblue" , linestyle = "--", marker='o')

    label = "VCL + k-Center (200)"
    acc_mean_file = np.nanmean(kcen_VCL, 1)
    plt.plot(np.arange(len(acc_mean_file))+1, acc_mean_file, label=label, color ="deepskyblue" , linestyle = "--", marker='d')

    label = 'JSD-VCL + Random (200)'
    acc_mean_file = np.nanmean(rand_JSD_VCL, 1)
    plt.plot(np.arange(len(acc_mean_file))+1, acc_mean_file, label=label, color ="orchid" , linestyle = "--", marker='o')

    label = "BD-VCL + Random (200)"
    acc_mean_file = np.nanmean(rand_BD_VCL, 1)
    plt.plot(np.arange(len(acc_mean_file))+1, acc_mean_file, label=label, color ="forestgreen" , linestyle = "--", marker='o')

    label = "coresetBD-VCL + Random (200)"
    acc_mean_file = np.nanmean(rand_coresetBD_VCL, 1)
    plt.plot(np.arange(len(acc_mean_file))+1, acc_mean_file, label=label, color ="r" , linestyle = "--", marker='o')

    label = "coresetBD-VCL + k-Center (200)"
    acc_mean_file = np.nanmean(kcen_coresetBD_VCL, 1)
    plt.plot(np.arange(len(acc_mean_file))+1, acc_mean_file, label=label, color ="firebrick" , linestyle = "--", marker='d')

    ax.set_xticks(range(1, len(acc_mean_file)+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('# tasks')
    ax.legend()
    plt.title("Average test set accuracy on Permuted MNIST - experimental results + coreset")
    plt.savefig("regularization_metric_with_coreset_results.jpg")

    ''' Split MNIST '''
    ## split_paper_replication_results ##
    x = np.arange(1,6)
    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['lines.linewidth'] = 0.75

    plt.clf()
    g, ax1 = plt.subplots(1, 6, figsize=(16,2), sharex=True,
                                                            sharey=True)
    h, ax2 = plt.subplots(1, 6, figsize=(16,2), sharex=True,
                                                            sharey=True)

    titles = ["Task 1 (0 or 1)", "Task 2 (2 or 3)", "Task 3 (4 or 5)", "Task 4 (6 or 7)", "Task 5 (8 or 9)", "Average"]

    for i in range(5):
        ax1[i].plot(x[i:], VCL_split[i:,i], color = "b", marker = "s")
        ax1[i].plot(x[i:], randVCL_split[i:,i],color = "royalblue", linestyle="--", marker = "o")
        ax1[i].plot(x[i:], rand_coreset_only_split[i:,i], color = "k", linestyle="--",marker = "o")
        ax1[i].plot(x[i:], kcenVCL_split[i:,i], color = "deepskyblue", linestyle = "--", marker = "d")
        ax1[i].plot(x[i:], kcen_coreset_only_split[i:,i], color = "k", linestyle = "--", marker = "d")


        ax2[i].plot(x[i:], VCL_split[i:,i],color = "b", marker = "s")
        ax2[i].plot(x[i:], randVCL_split[i:,i],color = "royalblue", linestyle = "--", marker = "o")
        ax2[i].plot(x[i:], rand_coreset_only_split[i:,i], color = "k", linestyle="--",marker = "o")
        ax2[i].plot(x[i:], kcenVCL_split[i:,i], color = "deepskyblue", linestyle = "--", marker = "d")
        ax2[i].plot(x[i:], kcen_coreset_only_split[i:,i], color = "k", linestyle = "--", marker = "d")

        # Add axis labels to each subplot
        ax1[i].set_title(titles[i])
        ax1[i].set_xlabel('Tasks')  # Set the x-axis label
        ax1[i].set_ylabel('Accuracy')  # Set the y-axis label
        ax2[i].set_title(titles[i])
        ax2[i].set_xlabel('Tasks')  # Set the x-axis label
        ax2[i].set_ylabel('Accuracy')  # Set the y-axis label

    ax1[-1].plot(x, np.nanmean(VCL_split,1), color = "b", marker = "s")
    ax1[-1].plot(x, np.nanmean(randVCL_split,1), color = "royalblue", linestyle="--",marker = "o")
    ax1[-1].plot(x, np.nanmean(rand_coreset_only_split,1), color = "k", linestyle="--", marker = "o")
    ax1[-1].plot(x, np.nanmean(kcenVCL_split,1), color = "deepskyblue", linestyle = "--", marker = "d")
    ax1[-1].plot(x, np.nanmean(kcen_coreset_only_split,1),color = "k", linestyle = "--", marker = "d")

    ax2[-1].plot(x, np.nanmean(VCL_split,1), color = "b", marker = "s")
    ax2[-1].plot(x, np.nanmean(randVCL_split,1), color = "royalblue", linestyle="--", marker = "o")
    ax2[-1].plot(x, np.nanmean(rand_coreset_only_split,1), color = "k", linestyle="--", marker = "o")
    ax2[-1].plot(x, np.nanmean(kcenVCL_split,1), color = "deepskyblue", linestyle = "--", marker = "d")
    ax2[-1].plot(x, np.nanmean(kcen_coreset_only_split,1),color = "k", linestyle = "--", marker = "d")

    # Add axis labels for the last subplot
    ax1[-1].set_title(titles[-1])
    ax1[-1].set_xlabel('Tasks')
    ax1[-1].set_ylabel('Accuracy')
    ax2[-1].set_title(titles[-1])
    ax2[-1].set_xlabel('Tasks')
    ax2[-1].set_ylabel('Accuracy')

    for i in range(6):
        ax1[i].set_xticks(np.arange(1, 6, 1))
        ax1[i].set_yticks(np.arange(0.25, 1.25, 0.25))
        ax2[i].set_xticks(np.arange(1, 6, 1))
        ax2[i].set_yticks(np.arange(0.85, 1.01, 0.05))
        plt.ylim(0.85,1.01)

    g.legend(   # The line objects
            labels=["VCL", "VCL + Rand","Coreset (Rand)", "VCL + kCen", "Coreset (kCen)"],   # The labels for each line
            loc="center right",   # Position of legend
            borderaxespad=0.3,    # Small spacing around legend box
            title="Methods:"  # Title for the legend
            )
    g.savefig("split_paper_replication_results (a).jpg")
    h.legend(   # The line objects
            labels=["VCL", "VCL + Rand","Coreset (Rand)", "VCL + kCen", "Coreset (kCen)"],   # The labels for each line
            loc="center right",   # Position of legend
            borderaxespad=0.3,    # Small spacing around legend box
            title="Methods:"  # Title for the legend
            )
    h.savefig("split_paper_replication_results (b).jpg")

    ## split experimental results ##
    x = np.arange(1,6)
    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['lines.linewidth'] = 0.75

    plt.clf()
    g, ax1 = plt.subplots(1, 6, figsize=(16,2), sharex=True,
                                                            sharey=True)
    h, ax2 = plt.subplots(1, 6, figsize=(16,2), sharex=True,
                                                            sharey=True)

    titles = ["Task 1 (0 or 1)", "Task 2 (2 or 3)", "Task 3 (4 or 5)", "Task 4 (6 or 7)", "Task 5 (8 or 9)", "Average"]

    for i in range(5):
        ax1[i].plot(x[i:], VCL_split[i:,i], color = "b", marker = "s")
        ax1[i].plot(x[i:], JSD_VCL_split[i:,i], color = "m",marker = "s")
        ax1[i].plot(x[i:], BD_VCL_split[i:,i], color = "g", marker = "s")

        ax2[i].plot(x[i:], VCL_split[i:,i], color = "b", marker = "s")
        ax2[i].plot(x[i:], JSD_VCL_split[i:,i], color = "m",marker = "s")
        ax2[i].plot(x[i:], BD_VCL_split[i:,i], color = "g", marker = "s")

        # Add axis labels to each subplot
        ax1[i].set_title(titles[i])
        ax1[i].set_xlabel('Tasks')  # Set the x-axis label
        ax1[i].set_ylabel('Accuracy')  # Set the y-axis label
        ax2[i].set_title(titles[i])
        ax2[i].set_xlabel('Tasks')  # Set the x-axis label
        ax2[i].set_ylabel('Accuracy')  # Set the y-axis label

    ax1[-1].plot(x, np.nanmean(VCL_split,1), color = "b", marker = "s")
    ax1[-1].plot(x, np.nanmean(JSD_VCL_split,1), color = "m",marker = "s")
    ax1[-1].plot(x, np.nanmean(BD_VCL_split,1), color = "g", marker = "s")

    ax2[-1].plot(x, np.nanmean(VCL_split,1), color = "b", marker = "s")
    ax2[-1].plot(x, np.nanmean(JSD_VCL_split,1), color = "m",marker = "s")
    ax2[-1].plot(x, np.nanmean(BD_VCL_split,1), color = "g", marker = "s")

    # Add axis labels for the last subplot
    ax1[-1].set_title(titles[-1])
    ax1[-1].set_xlabel('Tasks')
    ax1[-1].set_ylabel('Accuracy')
    ax2[-1].set_title(titles[-1])
    ax2[-1].set_xlabel('Tasks')
    ax2[-1].set_ylabel('Accuracy')

    for i in range(6):
        ax1[i].set_xticks(np.arange(1, 6, 1))
        ax1[i].set_yticks(np.arange(0.25, 1.25, 0.25))
        ax2[i].set_xticks(np.arange(1, 6, 1))
        ax2[i].set_yticks(np.arange(0.85, 1.01, 0.05))
        plt.ylim(0.85,1.01)

    g.legend(   # The line objects
            labels=["VCL", "JSD-VCL", "BD-VCL"],   # The labels for each line
            loc="center right",   # Position of legend
            borderaxespad=0.3,    # Small spacing around legend box
            title="Methods:"  # Title for the legend
            )
    g.savefig("split_experimental_results (a).jpg")
    h.legend(   # The line objects
            labels=["VCL", "JSD-VCL", "BD-VCL"],   # The labels for each line
            loc="center right",   # Position of legend
            borderaxespad=0.3,    # Small spacing around legend box
            title="Methods:"  # Title for the legend
            )
    h.savefig("split_experimental_results (b).jpg")

    ## split coreset results ##
    x = np.arange(1,6)
    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['lines.linewidth'] = 0.75

    plt.clf()
    g, ax1 = plt.subplots(1, 6, figsize=(20,2), sharex=True,
                                                            sharey=True)
    h, ax2 = plt.subplots(1, 6, figsize=(20,2), sharex=True,
                                                            sharey=True)

    titles = ["Task 1 (0 or 1)", "Task 2 (2 or 3)", "Task 3 (4 or 5)", "Task 4 (6 or 7)", "Task 5 (8 or 9)", "Average"]


    for i in range(5):
        ax1[i].plot(x[i:], kcenVCL_split[i:,i], color = "deepskyblue", linestyle="--", marker = "d")
        ax1[i].plot(x[i:], randVCL_split[i:,i],color = "royalblue", linestyle="--", marker = "o")
        ax1[i].plot(x[i:], kcen_JSD_VCL_split[i:,i], color = "violet", linestyle="--", marker = "d")
        ax1[i].plot(x[i:], rand_JSD_VCL_split[i:,i], color = "orchid", linestyle="--", marker = "o")
        ax1[i].plot(x[i:], kcen_BD_VCL_split[i:,i], color = "lawngreen",  linestyle="--", marker = "s")
        ax1[i].plot(x[i:], rand_BD_VCL_split[i:,i], color = "forestgreen", linestyle="--", marker = "o")
        ax1[i].plot(x[i:], kcen_coresetBD_VCL_split[i:,i], color = "firebrick", linestyle="--", marker = "d")
        ax1[i].plot(x[i:], rand_coresetBD_VCL_split[i:,i], color = "r", linestyle="--", marker = "o")

        ax2[i].plot(x[i:], kcenVCL_split[i:,i], color = "deepskyblue", linestyle="--", marker = "d")
        ax2[i].plot(x[i:], randVCL_split[i:,i],color = "royalblue", linestyle="--", marker = "o")
        ax2[i].plot(x[i:], kcen_JSD_VCL_split[i:,i], color = "violet", linestyle="--", marker = "d")
        ax2[i].plot(x[i:], rand_JSD_VCL_split[i:,i], color = "orchid", linestyle="--", marker = "o")
        ax2[i].plot(x[i:], kcen_BD_VCL_split[i:,i], color = "lawngreen",  linestyle="--", marker = "s")
        ax2[i].plot(x[i:], rand_BD_VCL_split[i:,i], color = "forestgreen", linestyle="--", marker = "o")
        ax2[i].plot(x[i:], kcen_coresetBD_VCL_split[i:,i], color = "firebrick", linestyle="--", marker = "d")
        ax2[i].plot(x[i:], rand_coresetBD_VCL_split[i:,i], color = "r", linestyle="--", marker = "o")


        # Add axis labels to each subplot
        ax1[i].set_title(titles[i])
        ax1[i].set_xlabel('Tasks')  # Set the x-axis label
        ax1[i].set_ylabel('Accuracy')  # Set the y-axis label
        ax2[i].set_title(titles[i])
        ax2[i].set_xlabel('Tasks')  # Set the x-axis label
        ax2[i].set_ylabel('Accuracy')  # Set the y-axis label

    ax1[-1].plot(x, np.nanmean(kcenVCL_split, 1), color = "deepskyblue", linestyle="--", marker = "d")
    ax1[-1].plot(x, np.nanmean(randVCL_split, 1),color = "royalblue", linestyle="--", marker = "o")
    ax1[-1].plot(x, np.nanmean(kcen_JSD_VCL_split,1), color = "violet", linestyle="--", marker = "d")
    ax1[-1].plot(x, np.nanmean(rand_JSD_VCL_split, 1), color = "orchid", linestyle="--", marker = "o")
    ax1[-1].plot(x, np.nanmean(kcen_BD_VCL_split, 1), color = "lawngreen", linestyle="--", marker = "d")
    ax1[-1].plot(x, np.nanmean(rand_BD_VCL_split, 1), color = "forestgreen", linestyle="--", marker = "o")
    ax1[-1].plot(x, np.nanmean(kcen_coresetBD_VCL_split, 1), color = "firebrick", linestyle="--", marker = "d")
    ax1[-1].plot(x, np.nanmean(rand_coresetBD_VCL_split,1), color = "r", linestyle="--", marker = "o")

    ax2[-1].plot(x, np.nanmean(kcenVCL_split, 1), color = "deepskyblue", linestyle="--", marker = "d")
    ax2[-1].plot(x, np.nanmean(randVCL_split, 1),color = "royalblue", linestyle="--", marker = "o")
    ax2[-1].plot(x, np.nanmean(kcen_JSD_VCL_split,1), color = "violet", linestyle="--", marker = "d")
    ax2[-1].plot(x, np.nanmean(rand_JSD_VCL_split, 1), color = "orchid", linestyle="--", marker = "o")
    ax2[-1].plot(x, np.nanmean(kcen_BD_VCL_split, 1), color = "lawngreen", linestyle="--", marker = "d")
    ax2[-1].plot(x, np.nanmean(rand_BD_VCL_split, 1), color = "forestgreen", linestyle="--", marker = "o")
    ax2[-1].plot(x, np.nanmean(kcen_coresetBD_VCL_split, 1), color = "firebrick", linestyle="--", marker = "d")
    ax2[-1].plot(x, np.nanmean(rand_coresetBD_VCL_split,1), color = "r", linestyle="--", marker = "o")

    # Add axis labels for the last subplot
    ax1[-1].set_title(titles[-1])
    ax1[-1].set_xlabel('Tasks')
    ax1[-1].set_ylabel('Accuracy')
    ax2[-1].set_title(titles[-1])
    ax2[-1].set_xlabel('Tasks')
    ax2[-1].set_ylabel('Accuracy')

    for i in range(6):
        ax1[i].set_xticks(np.arange(1, 6, 1))
        ax1[i].set_yticks(np.arange(0.25, 1.25, 0.25))
        ax2[i].set_xticks(np.arange(1, 6, 1))
        ax2[i].set_yticks(np.arange(0.95, 1.01, 0.01))
        plt.ylim(0.95,1.01)

    g.legend(   # The line objects
            labels=["VCL + kCen", "VCL + Rand", "JSD + kCen", 'JSD + Rand', "BD + kCen", "BD + Rand", "coresetBD + kCen", "coresetBD + Rand"],   # The labels for each line
            loc="center right",   # Position of legend
            borderaxespad=0.3,    # Small spacing around legend box
            title="Methods:"  # Title for the legend
            )
    g.savefig("split_coreset_results (a).jpg")
    h.legend(   # The line objects
            labels=["VCL + kCen", "VCL + Rand", "JSD + kCen", 'JSD + Rand', "BD + kCen", "BD + Rand", "coresetBD + kCen", "coresetBD + Rand"],   # The labels for each line
            loc="center right",   # Position of legend
            borderaxespad=0.3,    # Small spacing around legend box
            title="Methods:"  # Title for the legend
            )
    h.savefig("split_coreset_results (b).jpg")

        

