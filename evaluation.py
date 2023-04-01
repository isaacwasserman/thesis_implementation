# From https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/issues/2
# from https://aswali.github.io/WNet/
from glob import glob
import numpy as np
import tqdm
import os
import cv2
import skimage
from sklearn import metrics

def calculate_overlap(r1, r2):
    # intersection
    a = np.count_nonzero(r1 * r2)
    # union
    b = np.count_nonzero(r1 + r2)
    
    return a/b

def calculate_SC(segmentation1, id):
    if len(segmentation1.shape) > 2:
        segmentation1 = image_to_int_array(segmentation1)

    gt_image_paths = glob(f'BSDS500/gt/{id}*.csv')
    gt_images = []
    for gt_image_path in gt_image_paths:
        gt_images.append(np.loadtxt(gt_image_path, delimiter=',').astype(int))
        
    SCs = []
    for segmentation2 in gt_images:
        N = segmentation1.shape[0] * segmentation1.shape[1]
        
        maxcoverings_sum = 0
        
        # Sum over regions
        for label1 in np.unique(segmentation1):
            # region is where the segmentation has a specific label
            region1 = (segmentation1 == label1).astype(int) 
            # |R| is the size of nonzero elements as this the region size
            len_r = np.count_nonzero(region1) 
            max_overlap = 0
            # Calculate max overlap 
            for label2 in np.unique(segmentation2):
                # region is where the segmentation has a specific label
                region2 = (segmentation2 == label2).astype(int)
                # Calculate overlap
                overlap = calculate_overlap(region1, region2)
                max_overlap = max(max_overlap, overlap)
            
            maxcoverings_sum += (len_r * max_overlap)
            
        SCs.append( (1 / N) * maxcoverings_sum)
    return (np.mean(SCs), np.max(SCs))

def image_to_int_array(input_array):
    input_array_2d = np.zeros((input_array.shape[0], input_array.shape[1]))
    colors = np.unique(input_array.reshape(-1, input_array.shape[2]), axis=0)
    for i, color in enumerate(colors):
        input_array_2d[np.all(input_array == color, axis=2)] = i
    return input_array_2d.astype(int)

def calculate_VI(input_array, id):
    if len(input_array.shape) > 2:
        input_array = image_to_int_array(input_array)
    gt_image_paths = glob(f'BSDS500/gt/{id}*.csv')
    gt_images = []
    for gt_image_path in gt_image_paths:
        gt_images.append(np.loadtxt(gt_image_path, delimiter=',').astype(int))

    VIs = []
    for gt_image in gt_images:
        VI = skimage.metrics.variation_of_information(input_array, gt_image)
        VIs.append(VI)
    return (np.mean(VIs), np.min(VIs))

def calculate_PRI(input_array, id):
    if len(input_array.shape) > 2:
        input_array = image_to_int_array(input_array)
    gt_image_paths = glob(f'BSDS500/gt/{id}*.csv')
    gt_images = []
    for gt_image_path in gt_image_paths:
        gt_images.append(np.loadtxt(gt_image_path, delimiter=',').astype(int))

    VIs = []
    for gt_image in gt_images:
        VI = metrics.adjusted_rand_score(gt_image.flatten(), input_array.flatten())
        VIs.append(VI)
    return (np.mean(VIs), np.max(VIs))


def calculate_mIOU(input_array, id):
    os.makedirs('temp/gt', exist_ok=True)
    os.makedirs('temp/in', exist_ok=True)
    os.system(f'cp BSDS500/gt/{id}*.csv temp/gt')

    if len(input_array.shape) > 2:
        input_array = image_to_int_array(input_array)
        
    np.savetxt(f'temp/in/{id}.csv', input_array, delimiter=',')

    pred_path = 'temp/in'
    gt_path = 'temp/gt'

    user_input_flg = False
    resize_flg = False
    no_background_flg = False

    args_input = pred_path
    args_gt = gt_path
    args_bsd500 = True
    args_mode = 1

    input_list = sorted(glob(args_input + '/*'))

    miou_list = []
    max_miou_list = []
    categorical_miou_array = np.zeros( (21, 2) )

    for input_csv in input_list:
        if True:
            raw_input_array = np.loadtxt(input_csv, delimiter=',')
            if user_input_flg:
                input_csv = input_csv[:-3] + "csv"
                gt_arrays = np.loadtxt(args_gt + input_csv[-16:], delimiter=',')
            elif args_bsd500:
                no_background_flg = False
                gt_arrays = []
                for i in range(100):
                    fname = args_gt + "/" + input_csv.split("/")[-1][:-4] + "-" + str(i) + ".csv"
                    if not os.path.exists(fname):
                        break
                    gt_arrays.append( np.loadtxt(fname, delimiter=',') )
                if args_mode == 2:
                    gt_arrays = gt_arrays[ np.argmax( np.array([ len(np.unique(g)) for g in gt_arrays ]) ) ]
                elif args_mode == 3:
                    gt_arrays = gt_arrays[ np.argmin( np.array([ len(np.unique(g)) for g in gt_arrays ]) ) ]
                gt_arrays = np.array(gt_arrays)
            else:
                gt_arrays = cv2.imread(args_gt + input_csv[-16:-3]+"png", -1)
            if resize_flg:
                input_array = cv2.resize( raw_input_array, (gt_arrays.shape[1],gt_arrays.shape[0]) , interpolation = cv2.INTER_NEAREST  )
            else:
                input_array = raw_input_array

        if len(gt_arrays.shape) == 2:
            gt_arrays = [gt_arrays]
        
        miou_per_gt_segmentation = []
        for gt_array in gt_arrays:
            miou_for_each_class = []
            label_list = np.unique(gt_array)

            # gt_mask is 0 where gt label is 0 (background) and 1 where gt label is not 0 (foreground)
            gg = np.zeros(gt_array.shape)
            gt_mask = np.where(gt_array > 0, 1, gg)

            # determinant array is input array but with background discarded (set to 0)
            determinant_array = gt_mask * input_array
            # label_list is the list of labels in the input array (range(k))
            label_list = np.unique(gt_array)


            gt_array_1d = gt_array.reshape((gt_array.shape[0]*gt_array.shape[1])) # 1d array of gt labels
            input_array_1d = input_array.reshape((input_array.shape[0]*input_array.shape[1])) # 1d array of input labels

            # For each class in range k
            for l in label_list:
                inds = np.where( gt_array_1d == l )[0] # indices of gt where label is l
                pred_labels = input_array_1d[ inds ] # predictions at those indices
                u_pred_labels = np.unique(pred_labels) # unique predictions at those indices
                hists = [ np.sum(pred_labels == u) for u in u_pred_labels ] # frequency of each unique prediction at those indices
                fractions = [ len(inds) + np.sum(input_array_1d == u) - np.sum(pred_labels == u) for u in u_pred_labels ] # (total number of pixels in gt with label l) + (total number of pixels in input with label u) - (total number of pixels in input with label u and gt label l)
                mious = hists / np.array(fractions,dtype='float')
                miou_list.append( np.max(mious) )
                miou_for_each_class.append( np.max(mious) )
            miou_per_gt_segmentation.append( np.mean(miou_for_each_class) )
        max_miou_list.append( np.max(miou_per_gt_segmentation) )
        

    average_mIOU = sum(miou_list) / float(len(miou_list))
    max_mIOU = sum(max_miou_list) / float(len(max_miou_list))

    os.system('rm -rf temp')

    return average_mIOU, max_mIOU

def calculate_mIOU_batch(gt_path, pred_path):
    user_input_flg = False
    resize_flg = False
    no_background_flg = False

    args_input = pred_path
    args_gt = gt_path
    args_bsd500 = True
    args_mode = 1

    input_list = sorted(glob(args_input + '/*'))

    miou_list = []
    max_miou_list = []
    categorical_miou_array = np.zeros( (21, 2) )

    for input_csv in tqdm.tqdm(input_list):
        if True:
            raw_input_array = np.loadtxt(input_csv, delimiter=',')
            if user_input_flg:
                input_csv = input_csv[:-3] + "csv"
                gt_arrays = np.loadtxt(args_gt + input_csv[-16:], delimiter=',')
            elif args_bsd500:
                no_background_flg = False
                gt_arrays = []
                for i in range(100):
                    fname = args_gt + "/" + input_csv.split("/")[-1][:-4] + "-" + str(i) + ".csv"
                    if not os.path.exists(fname):
                        break
                    gt_arrays.append( np.loadtxt(fname, delimiter=',') )
                if args_mode == 2:
                    gt_arrays = gt_arrays[ np.argmax( np.array([ len(np.unique(g)) for g in gt_arrays ]) ) ]
                elif args_mode == 3:
                    gt_arrays = gt_arrays[ np.argmin( np.array([ len(np.unique(g)) for g in gt_arrays ]) ) ]
                gt_arrays = np.array(gt_arrays)
            else:
                gt_arrays = cv2.imread(args_gt + input_csv[-16:-3]+"png", -1)
            if resize_flg:
                input_array = cv2.resize( raw_input_array, (gt_arrays.shape[1],gt_arrays.shape[0]) , interpolation = cv2.INTER_NEAREST  )
            else:
                input_array = raw_input_array

        if len(gt_arrays.shape) == 2:
            gt_arrays = [gt_arrays]
        
        miou_per_gt_segmentation = []
        for gt_array in gt_arrays:
            miou_for_each_class = []
            label_list = np.unique(gt_array)

            # gt_mask is 0 where gt label is 0 (background) and 1 where gt label is not 0 (foreground)
            gg = np.zeros(gt_array.shape)
            gt_mask = np.where(gt_array > 0, 1, gg)

            # determinant array is input array but with background discarded (set to 0)
            determinant_array = gt_mask * input_array
            # label_list is the list of labels in the input array (range(k))
            label_list = np.unique(gt_array)


            gt_array_1d = gt_array.reshape((gt_array.shape[0]*gt_array.shape[1])) # 1d array of gt labels
            input_array_1d = input_array.reshape((input_array.shape[0]*input_array.shape[1])) # 1d array of input labels

            # For each class in range k
            for l in label_list:
                inds = np.where( gt_array_1d == l )[0] # indices of gt where label is l
                pred_labels = input_array_1d[ inds ] # predictions at those indices
                u_pred_labels = np.unique(pred_labels) # unique predictions at those indices
                hists = [ np.sum(pred_labels == u) for u in u_pred_labels ] # frequency of each unique prediction at those indices
                fractions = [ len(inds) + np.sum(input_array_1d == u) - np.sum(pred_labels == u) for u in u_pred_labels ] # (total number of pixels in gt with label l) + (total number of pixels in input with label u) - (total number of pixels in input with label u and gt label l)
                mious = hists / np.array(fractions,dtype='float')
                miou_list.append( np.max(mious) )
                miou_for_each_class.append( np.max(mious) )
            miou_per_gt_segmentation.append( np.mean(miou_for_each_class) )
        max_miou_list.append( np.max(miou_per_gt_segmentation) )
        

    average_mIOU = sum(miou_list) / float(len(miou_list))
    max_mIOU = sum(max_miou_list) / float(len(max_miou_list))
    return average_mIOU, max_mIOU