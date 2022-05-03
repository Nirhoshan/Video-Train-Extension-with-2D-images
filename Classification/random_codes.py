import os
import cv2
import matplotlib.pyplot as plt


def rename_files():
    type = 'mtf'
    pltform = 'Sta'
    path_in = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/data/Gasf-Gadf-Mtf-percentile-mapping/' + type + '/' + pltform

    for v in range(20):
        for t in range(100):
            os.rename(path_in + '/' + type + '_' + pltform + '_vid' + str(v + 1) + '_' + str(t + 1) + '.png',
                      path_in + '/' + pltform + '_vid' + str(v + 1) + '_' + str(t + 1
                                                                                ) + '.png')
    return


# rename some files synthesized by Nirohshan
def rename_files_64_by_64_synth():
    vid = 'Youtube'
    gpu =3

    #path_in = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/data/data_64_by_64/gasf/synth/complete_vid_seperated/' + '/' + vid

    path_in = 'C:/Users/Nirho/Desktop/gasf-dl/gasf_60_generated/' + vid
    path_out = 'C:/Users/Nirho/Desktop/gasf-dl/dl_clf_60/' + vid + '/synthesized'
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    for v in range(20):
        # if v==0:
        #     continue
        vid_path_in = path_in + '/vid' + str(v + 1) + '/vid' + str(v + 1) + '/generated' + str(gpu)
        vid_path_out = path_out + '/vid' + str(v + 1)
        if not os.path.exists(vid_path_out):
            os.makedirs(vid_path_out)

        list_of_files = os.listdir(vid_path_in)
        if '.DS_Store' in list_of_files:
            list_of_files.remove('.DS_Store')

        for f_id, f in enumerate(list_of_files):
            os.rename(path_in + '/vid' + str(v + 1) + '/vid' + str(v + 1) + '/generated' + str(gpu) + '/' + f,
                      vid_path_out + '/' + vid + '_' + str(f_id + 1) + '.csv')
    return


# rename some files synthesized by Nirohshan
def rename_files_64_by_64_actual():
    platform = 'Netflix'

    #path_in = 'C:/Users/Nirho/Desktop/gasf/gasf-csv/gasf_csv_bin4_40/' + platform + '/actual/' + platform
    path_in = 'C:/Users/Nirho/Desktop/mtf/dl_n/' + platform
    #path_out = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/data/gasf_increased_bin_size/' + platform + '/actual/bin_1'
    path_out = 'C:/Users/Nirho/Desktop/mtf/dl_clf_mtf/' + platform + '/actual'
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    for v in range(20):
        for t in range(100):
            vid_path_in = path_in + '/mtf_' + platform + '_vid' + str(v + 1) + '_' + str(t + 1) + '.csv'
            vid_path_out = path_out + '/' + platform + '_vid' + str(v + 1) + '_' + str(t + 1) + '.csv'

            os.rename(vid_path_in, vid_path_out)
    return


# down sample the image
def downsample_img():
    path_in = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/data/gasf_gadf_mtf' + '/gasf/Netflix/Netflix_vid1_12.png'

    im = cv2.imread(path_in)
    resized = cv2.resize(im, (100, 100), interpolation=cv2.INTER_AREA)

    plt.imshow(im)
    plt.show()
    plt.close()

    plt.imshow(resized)
    plt.show()
    plt.close()

    print(1)

    return


def remov_margin():
    # path_in = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/data/data_64_by_64/gasf/actual/Netflix/vid1/gasf_Netflix_vid1_1.png'
    path_in = '/Users/ckat9988/Documents/Research/CS_work_new/CSIRO_clf/data/data_64_by_64/gasf/synth/Netflix/vid1/generated/gasf_Netflix_1.png'

    im = cv2.imread(path_in)
    plt.imshow(im)
    plt.show()

    im = im[7:57, 8:58, :]
    plt.imshow(im)
    plt.show()

    return


def main():
    # rename_files()
    # downsample_img()

    # remov_margin()
    rename_files_64_by_64_synth()
    #rename_files_64_by_64_actual()

    return


if __name__ == main():
    main()
