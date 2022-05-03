import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# bin_sizes = [1, 2, 4, 6, 8, 10]
# x_labels = ['1', '2', ' 4', ' 6', '8', '10']
# path_in = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/results/bin_size_analysis/Netflix/kernel_size_8'
#
# acc = []
# for b in bin_sizes:
#     agg_path_in = path_in + '/acc_aggreg_' + str(b) + '.csv'
#     acc.append(np.mean(pd.read_csv(agg_path_in).values * 100))
#
# fig, ax = plt.subplots(1, 1)
# plt.bar(np.arange(len(acc)), acc)
# ax.set_xticks(np.arange(len(bin_sizes)))
# ax.set_xticklabels(x_labels)
# plt.xlabel('Bin aggregation')
# plt.ylabel('Accuracy')
# plt.ylim(0, 100)
# plt.show()


def analyse_diff_bin_impact(platform):
    path_in = 'C:/Users/Nirho/Desktop/gasf-dl/dl_clf_all/results/diff_bin_size_analysis_with_synth_data_reduced_act_traces/gasf/split_the_traces_in_a_video/' + platform
    #path_in = 'C:/Users/Nirho/Desktop/gasf/data_clf_all/results/diff_bin_size_analysis_with_synth_data_reduced_act_traces/gasf/split_the_traces_in_a_video/' + platform
    bins = [4]  # , 8]
    train_data_size=20
    all_b_acc = []
    for bin in bins:
        bin_path = path_in + '/bin_' + str(bin)

        all_r_acc = []
        for r in range(3):
            rand_path = bin_path +'/train_data_size_'+str(train_data_size)+ '/Random_set_' + str(r)

            all_r_acc.append(pd.read_csv(rand_path + '/acc.csv').values * 100)

        all_b_acc.append(all_r_acc)

    all_b_acc = np.asarray(all_b_acc).squeeze()
    all_b_acc = np.mean(all_b_acc, axis=0)

    x_labels = np.arange(0, 301, 30).astype(str)
    x = np.arange(len(all_b_acc))
    fig, ax = plt.subplots(1, 1)

    plt.plot(x, all_b_acc, c='r', label='Bin 4', linestyle='--')
    # plt.plot(x, all_b_acc[1], c='b', label='Bin 8', linestyle='--')

    plt.grid()
    plt.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    plt.xlabel('Synthesis data per video')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100)
    plt.show()

    print(1)

    return

def all_train_data(platform):
    path_in = 'C:/Users/Nirho/Desktop/gasf-dl/dl_clf_synth/results/diff_bin_size_analysis_with_synth_data_reduced_act_traces/gasf/split_the_traces_in_a_video/' + platform
    #path_in = 'C:/Users/Nirho/Desktop/gasf/data_clf_all/results/diff_bin_size_analysis_with_synth_data_reduced_act_traces/gasf/split_the_traces_in_a_video/' + platform
    bins = [4]  # , 8]

    all_b_acc = []
    for bin in bins:
        bin_path = path_in + '/bin_' + str(bin)

        all_r_acc = []
        all_0=[]
        all_30=[]
        all_60=[]
        all_90=[]
        all_120=[]
        all_150=[]
        all_180 = []
        all_210=[]
        all_240=[]
        all_270=[]
        all_300=[]
        max_train_data_size=80

        for train_data_size in range(20,max_train_data_size+1,20):
            all_r_acc = []
            for r in range(3):
                rand_path = bin_path +'/train_data_size_'+str(train_data_size)+ '/Random_set_' + str(r)
                #print(rand_path)
                all_r_acc.append(pd.read_csv(rand_path + '/acc.csv').values * 100)

            #print(all_r_acc)
            all_r_acc = np.asarray(all_r_acc).squeeze()
            #print(all_r_acc)

            all_r_acc = np.mean(all_r_acc, axis=0)
            #print(all_r_acc)
            #all_0.append(all_r_acc[0][0][0])
            #print(all_0)
            #all_5.append(all_r_acc[0][0][1])
            #all_10.append(all_r_acc[0][0][2])
            #all_15.append(all_r_acc[0][0][3])
            #all_20.append(all_r_acc[0][0][4])
            #all_25.append(all_r_acc[0][0][5])
            #all_30.append(all_r_acc[0][0][6])

            #all_0.append(all_r_acc[0])
            # print(all_0)
            all_30.append(all_r_acc[0])
            all_60.append(all_r_acc[1])
            all_90.append(all_r_acc[2])
            all_120.append(all_r_acc[3])
            all_150.append(all_r_acc[4])
            all_180.append(all_r_acc[5])
            all_210.append(all_r_acc[6])
            all_240.append(all_r_acc[7])
            all_270.append(all_r_acc[8])
            all_300.append(all_r_acc[9])
    #all_b_acc = np.asarray(all_b_acc).squeeze()
    #all_b_acc = np.mean(all_b_acc, axis=0)

    x_labels = np.arange(20, max_train_data_size+1,20).astype(str)
    x = np.arange(len(all_30))
    #print(all_0)
    fig, ax = plt.subplots(1, 1)

    #plt.plot(x, all_0, label='Bin 4', linestyle='--')
    # plt.plot(x, all_b_acc[1], c='b', label='Bin 8', linestyle='--')



    #plt.plot(x, all_0, label='actual', linestyle='--')
    plt.plot(x, all_30, label='synth 30', linestyle='--')
    plt.plot(x, all_60, label='synth 60', linestyle='--')
    plt.plot(x, all_90, label='synth 90', linestyle='--')
    plt.plot(x, all_120, label='synth 120', linestyle='--')
    plt.plot(x, all_150, label='synth 150', linestyle='--')
    plt.plot(x, all_180, label='synth 180', linestyle='--')
    plt.plot(x, all_210, label='synth 210', linestyle='--')
    plt.plot(x, all_240, label='synth 240', linestyle='--')
    plt.plot(x, all_270, label='synth 270', linestyle='--')
    plt.plot(x, all_300, label='synth 300', linestyle='--')
    plt.grid()
    plt.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    plt.xlabel('Per training data size')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100)
    plt.show()

    print(1)

    return



def analyse_diff_coff_mat(platform):
    path_in = 'C:/Users/Nirho/Desktop/gasf/data_clf/results/diff_bin_size_analysis_with_synth_data_reduced_act_traces/gasf/split_the_traces_in_a_video/' + platform
    bins = [4]  # , 8]

    all_b_coff = []
    for bin in bins:
        bin_path = path_in + '/bin_' + str(bin)

        all_r_coff = []
        for r in range(5):
            rand_path = bin_path + '/Random_set_' + str(r)

            all_s_coff = []
            for s in range(0, 501, 100):
                synth_coff = rand_path + '/Synthesized_size_' + str(s) + '.csv'
                all_s_coff.append(pd.read_csv(synth_coff, header=None).values)
            all_r_coff.append(all_s_coff)
        all_b_coff.append(all_r_coff)

    all_b_coff = np.asarray(all_b_coff)
    all_b_coff = all_b_coff.squeeze()
    all_b_coff = np.mean(all_b_coff, axis=0)

    x = np.arange(0, 20)
    x_labels = np.arange(1, 21).astype(str)

    title_lbl = np.arange(0, 501, 100).astype(str)
    for s in range(6):
        fig, ax = plt.subplots(1, 1)
        plt.imshow(all_b_coff[s])

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('synth data ' + str(title_lbl[s]))

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(x)
        ax.set_yticklabels(x_labels)

        plt.show()

        print(1)

    return


def analyse_bin():
    path_in = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/results/bin_size_analysis/Stan'

    bins = [1, 4, 6, 8]

    all_acc = []
    for b in bins:
        bin_path = path_in + '/acc_aggreg_' + str(b) + '.csv'
        acc = pd.read_csv(bin_path).values
        all_acc.append(acc)

    all_acc = np.asarray(all_acc)
    mean_acc = np.mean(all_acc, axis=0)
    sd_acc = np.std(all_acc, axis=0)
    print(mean_acc)
    print(sd_acc)

    return


# analyse_bin()

platform = 'Youtube'
#analyse_diff_bin_impact(platform)
#analyse_diff_coff_mat(platform)
all_train_data(platform)