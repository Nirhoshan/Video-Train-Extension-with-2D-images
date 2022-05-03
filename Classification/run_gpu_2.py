# keras based classification models with separate classifiers.
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle

import os
import pandas as pd
import argparse

break_point = 20

GPU = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

platforms = ['Youtube', 'Stan', 'Netflix']


# load YouTube, Stan and Netflix data for a given image type
def load_data_actual(im_type, platform, d_type, bin_size):
    if GPU:
        # path_in = '/share/home/ckat9988/CSU_work_extended/CSIRO_data_clf/data/data_64_by_64/' + im_type + '/' + d_type
        #path_in = '/share/home/ckat9988/CSU_work_extended/CSIRO_data_clf/data/csv_format/' + im_type + '_increased_format'
        path_in = 'dl_clf_20'
    else:
        #path_in = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/data/' + im_type + '_increased_bin_size'
        path_in = 'C:/Users/Nirho/Desktop/gasf/gasf-csv/for_clf_80_gamma'
    print(path_in)
    all_platforms = []

    gt_train_vids = []
    gt_test_vids = []

    for p_ind, p in enumerate(platforms):
        if p_ind != platform:
            continue
        print(p)
        #platform_path = path_in + '/' + p + '/' + d_type + '/bin_' + str(bin_size)
        platform_path = path_in + '/' + p + '/' + d_type
        all_vids = []

        gt_train_vid_for_platrom = []
        gt_test_vid_for_platrom = []

        for v in range(20):
            #if v == 7 or v==10:
             #   continue
            print(v)
            all_traces = []
            gt_all_traces_train = []
            gt_all_traces_test = []
            #for t in range(100):
            for t in range(100):
                trace_path = platform_path + '/' + p + '_vid' + str(v + 1) + '_' + str(t + 1) + '.csv'
                csv_data = pd.read_csv(trace_path, header=None).values
                csv_data = 0.5 * csv_data + 0.5
                A = 1
                gamma = 0.25
                csv_data = A * np.power(csv_data, gamma)
                all_traces.append(csv_data)
                if t < 80:
                    gt_all_traces_train.append(v)
                else:
                    gt_all_traces_test.append(v)

            all_vids.append(all_traces)
            gt_train_vid_for_platrom.append(gt_all_traces_train)
            gt_test_vid_for_platrom.append(gt_all_traces_test)

            if v == break_point:
                break

        gt_train_vids.append(gt_train_vid_for_platrom)
        gt_test_vids.append(gt_test_vid_for_platrom)

        all_platforms.append(all_vids)

    return all_platforms, np.asarray(gt_train_vids), np.asarray(gt_test_vids)


# load synthesized data
def load_synth_data(im_type, platform, d_type, bin_size):
    if GPU:
        # path_in = '/share/home/ckat9988/CSU_work_extended/CSIRO_data_clf/data/data_64_by_64/' + im_type + '/' + d_type
        #path_in = '/share/home/ckat9988/CSU_work_extended/CSIRO_data_clf/data/csv_format/' + im_type + '_increased_format'
        path_in = 'dl_clf_20'
    else:
        #path_in = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/data/' + im_type + '_increased_bin_size'
        path_in = 'C:/Users/Nirho/Desktop/gasf/gasf-csv/for_clf'
    print(path_in)
    all_platforms = []
    gt_train_vids = []

    for p_ind, p in enumerate(platforms):
        if p_ind != platform:
            continue
        print(p)

        platform_path = path_in + '/' + p + '/' + d_type

        all_vids = []
        gt_train_vid_for_platrom = []

        for v in range(20):
            #if v == 7 or v==10:
            #    continue
            vid_path = platform_path + '/vid' + str(v + 1)

            # select the first 500 traces
            all_traces = []
            gt_train_all_traces = []
            for t in range(400):
                print(t)
                trace_path = vid_path + '/' + p + '_' + str(t + 1) + '.csv'
                csv_data = pd.read_csv(trace_path, header=None).values
                #csv_data = 0.5 * csv_data + 0.5
                #A = 1
                #gamma = 0.25
                #csv_data = A * np.power(csv_data, gamma)
                gt_train_all_traces.append(v)
                all_traces.append(csv_data)

            all_vids.append(all_traces)
            gt_train_vid_for_platrom.append(gt_train_all_traces)

            if v == break_point:
                break

        gt_train_vids.append(gt_train_vid_for_platrom)
        all_platforms.append(all_vids)

    return all_platforms, np.asarray(gt_train_vids)


def define_model_1(train_data, output_units):
    print('Create model')
    # example of a 3-block vgg style architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3])))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))

    # example output part of the model
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(output_units, activation='softmax'))

    return model


def define_model_2(train_data, output_units):
    print('Create model')
    # example of a 3-block vgg style architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(train_data.shape[1], train_data.shape[2], 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2, 2)))

    # example output part of the model
    model.add(Flatten())
    model.add(Dense(128, activation='relu', ))
    model.add(Dense(64, activation='relu', ))
    model.add(Dense(output_units, activation='softmax'))

    return model


# # build CNN classifier and run the classification
# def run_paltform_clf(train_data, test_data, gt):
#     train_data = train_data.reshape([train_data.shape[0] * train_data.shape[1] * train_data.shape[2],
#                                      train_data.shape[3], train_data.shape[4], train_data.shape[5]])
#     test_data = test_data.reshape([test_data.shape[0] * test_data.shape[1] * test_data.shape[2],
#                                    test_data.shape[3], test_data.shape[4], test_data.shape[5]])
#
#     # create random index array for the given platform
#     # ind_arr = np.arange(train_data.shape[0])
#     x_train, y_train = shuffle(train_data, gt[0], random_state=0)
#     y_train = to_categorical(y_train)
#     x_test = test_data
#     y_test = to_categorical(gt[1])
#
#     # example of a 3-block vgg style architecture
#
#     model = define_model_2(train_data, output_units=3)
#
#     # opt = SGD(lr=0.001, momentum=0.9)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     print('train model')
#     history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=0)
#     # evaluate model
#     _, acc = model.evaluate(x_test, y_test, verbose=0)
#     print('> %.3f' % (acc * 100.0))
#     # learning curves
#
#     return


def run_vid_clf(x_train, y_train, x_test, y_test, platform, synth_size, random_state, bin_size, train_data_size):
    print('Create model')
    # example of a 3-block vgg style architecture
    y_train = to_categorical(y_train)

    model = define_model_2(x_train, output_units=break_point)

    # opt = SGD(lr=0.001, momentum=0.9)
    optimizer = SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model)
    model.fit(x_train, y_train, epochs=80, batch_size=8, verbose=True)
    # epochs : 20(init)-->30-->30-->20
    # batchsize : 32(init)-->32-->16-->16

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_pred, y_test)

    # precsion = average_precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred, average='binary')

    print('=================')
    print(platforms[platform])
    print('Random_set ' + str(random_state) + ' Synth size ' + str(synth_size) + '  Accuracy ' + str(accuracy))
    print('=================')
    # print('train model')
    # history = model.fit(x_train, y_train, epochs=32, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    # # evaluate model
    # _, acc = model.evaluate(x_test, y_test, verbose=0)
    # print('> %.3f' % (acc * 100.0))
    # # learning curves

    #path_to_save = '/share/home/ckat9988/CSU_work_extended/CSIRO_data_clf/results/diff_bin_size_analysis_with_synth_data_reduced_act_traces/gasf/' + '/' + 'split_the_traces_in_a_video' + '/' + \
    #               platforms[platform] + '/bin_' + str(bin_size) + '/Random_set_' + str(random_state)
    path_to_save = 'dl_clf_all/results/diff_bin_size_analysis_with_synth_data_reduced_act_traces/gasf/' + '/' + 'split_the_traces_in_a_video' + '/' + \
                   platforms[platform] + '/bin_' + str(bin_size) + '/train_data_size_' + str(train_data_size) + '/Random_set_' + str(random_state)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    confuse_matrix = confusion_matrix(y_test, y_pred)
    data = pd.DataFrame(data=confuse_matrix)
    data.to_csv(path_to_save + '/Synthesized_size_' + str(synth_size) + '.csv', index=False, header=False)

    return accuracy


def main():
    # parser = argparse.ArgumentParser(description="An argparse example")
    #
    # parser.add_argument('--p', help='Platform')
    # parser.add_argument('--g', help='gpu')
    #
    # args = parser.parse_args()

    # The following do not work:
    # print(args.foo-bar)
    # print(args.foo_bar)

    # But this works:
    # print(getattr(args, 'foo-bar'))

    im_type = 'gasf'
    platform = 0
    bins = [4]
    train_data_size = 20

    # run for different bin sizes
    for bin in bins:
        # load dat
        print('load data')
        feat_data_actual, gt_train_actual, gt_test_actual = load_data_actual(im_type=im_type,
                                                                             platform=platform,
                                                                             d_type='actual',
                                                                             bin_size=bin)

        print('load synth data')
        feat_data_synth, gt_train_synth = load_synth_data(im_type=im_type, platform=platform,
                                                          d_type='synthesized',
                                                          bin_size=bin)

        print('convert data to numpy array')
        feat_data_actual = np.asarray(feat_data_actual)
        feat_data_synth = np.asarray(feat_data_synth)
        print(feat_data_actual.shape)
        print(feat_data_synth.shape)

        # split data to train and test sets

        # run through CNN
        # random suffling of the

        # run_paltform_clf(X_train_actual, X_test_actual, gt_platform_actual)
        paltform_count = 0
        for p in range(3):
            if p != platform:
                continue

            X_train_actual = feat_data_actual[paltform_count, :, :train_data_size, :, :]
            print('========')
            print(X_train_actual.shape)
            print('========')
            X_train_actual = np.reshape(X_train_actual, (
                X_train_actual.shape[0] * X_train_actual.shape[1], X_train_actual.shape[2], X_train_actual.shape[3]))

            X_test_actual = feat_data_actual[paltform_count, :, 80:, :, :]
            X_test_actual = np.reshape(X_test_actual, (
                X_test_actual.shape[0] * X_test_actual.shape[1], X_test_actual.shape[2], X_test_actual.shape[3]))

            y_train_actual = gt_train_actual[paltform_count, :, :train_data_size]
            y_train_actual = np.reshape(y_train_actual, (y_train_actual.shape[0] * y_train_actual.shape[1]))

            y_test_actual = gt_test_actual[paltform_count, :, :]
            y_test_actual = np.reshape(y_test_actual, (y_test_actual.shape[0] * y_test_actual.shape[1]))

            all_rands = []
            for r in range(3):

                all_synth_acc = []
                for synth_size in range(0, 301, 30):
                    X_train_synth = feat_data_synth[paltform_count, :, :synth_size, :, :]
                    X_train_synth = np.reshape(X_train_synth, (
                        X_train_synth.shape[0] * X_train_synth.shape[1], X_train_synth.shape[2],
                        X_train_synth.shape[3]))

                    y_train_synth = gt_train_synth[paltform_count, :, :synth_size]
                    y_train_synth = np.reshape(y_train_synth, (y_train_synth.shape[0] * y_train_synth.shape[1]))

                    # concat synth train X and y data
                    X_train = np.concatenate([X_train_actual, X_train_synth], axis=0)
                    y_train = np.concatenate([y_train_actual, y_train_synth])

                    # X_train = X_train_synth
                    # y_train = y_train_synth

                    x_train, y_train = shuffle(X_train, y_train, random_state=r)

                    acc = run_vid_clf(x_train=x_train,
                                      y_train=y_train,
                                      x_test=X_test_actual,
                                      y_test=y_test_actual,
                                      platform=platform,
                                      synth_size=synth_size,
                                      random_state=r,
                                      bin_size=bin,
                                      train_data_size=train_data_size)
                    all_synth_acc.append(acc)

                # store the data
                columns = list(np.arange(0, 301, 30).astype(str))
                data = np.asarray(all_synth_acc).reshape([1, -1])
                df_acc = pd.DataFrame(columns=columns,
                                      data=data)
                path_to_save = 'dl_clf_all/results/diff_bin_size_analysis_with_synth_data_reduced_act_traces/gasf/' + '/' + 'split_the_traces_in_a_video' + '/' + \
                               platforms[platform] + '/bin_' + str(bin) + '/train_data_size_' + str(train_data_size) + '/Random_set_' + str(r)
                df_acc.to_csv(path_to_save + '/acc' + '.csv', index=False)

            paltform_count += 1
            # break
        # break
    return


if __name__ == main():
    main()
