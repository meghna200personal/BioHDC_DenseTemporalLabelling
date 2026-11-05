import numpy as np
from numpy.random import default_rng

# from multiprocessing import Pool
# from multiprocessing import shared_memory
# import signal
# import os

rng = default_rng()


def rand_sign(hv):
    '''
        signs an HV but sets 0 values to be -1 or 1 randomly. Necessary because np.sign returns 0 when element == 0.
    '''
    c = np.array(hv)
    zero_elems = np.where(c==0)
    c[zero_elems] = np.random.choice([-1,1], len(zero_elems[0]))
    return np.array(np.sign(c), dtype=np.int8)

def calculate_quantization_parameters(train_data, num_quantiles, verbose=True):
    """
    Discretizes the range of feature values into quantiles and calculates midpoints.
    
    Args:
        train_data (np.array): Training data matrix of shape (n_samples, n_features).
        num_quantiles (int): Number of bins to create.
        verbose (bool): Whether to print status updates.
        
    Returns:
        tuple: (quantile_edges, quantile_midpoints)
    """
    n_samples, n_features = train_data.shape
    
    # 1. Calculate boundaries per feature
    mins = np.min(train_data, axis=0)  # Shape: (n_features,)
    maxes = np.max(train_data, axis=0) # Shape: (n_features,)
    feature_ranges = maxes - mins      # Shape: (n_features,)
    
    # 2. Vectorized calculation of edges
    # We create a scale from 0 to 1 and broadcast it across the feature ranges
    steps = np.linspace(0, 1, num_quantiles).reshape(-1, 1) # Shape: (num_quantiles, 1)
    
    # Broadcating (num_quantiles, 1) * (n_features,) -> (num_quantiles, n_features)
    quantile_edges = mins + (steps * feature_ranges)
    
    # 3. Calculate midpoints
    # The interval width for each feature
    interval_width = feature_ranges / (num_quantiles - 1)
    
    # Midpoints are the edges (excluding the last one) plus half the interval width
    # Shape becomes (num_features, num_quantiles - 1) due to the transpose
    quantile_midpoints = (quantile_edges[:-1, :] + (interval_width / 2)).T

    if verbose:
        print(f"Quantization complete for {n_features} features.")
        print(f"Edges Shape: {quantile_edges.shape}")
        print(f"Midpoints Shape: {quantile_midpoints.shape}")

    return quantile_edges, quantile_midpoints

def generate_CIM(num_quantiles, dimensionality):
    '''
        Generates a Continuous Item Memory (CIM) containing num_quantiles HVs of dimension dimensionality
    '''

    #flips bits such that first and last HVs are orthogonal.
    num_bits_to_flip = np.floor(dimensionality/(num_quantiles*2))
    cim = [np.random.choice([-1,1], size=(dimensionality)).astype(int)]

    order_bits_to_flip = np.arange(dimensionality)
    np.random.shuffle(order_bits_to_flip)
    for i in range(1, num_quantiles):
        cim.append(np.copy(cim[-1]))
        for j in range(int((i-1)*num_bits_to_flip), int(i*num_bits_to_flip)):
            cim[i][order_bits_to_flip[j]] = int(-1 * cim[i][order_bits_to_flip[j]])
    
    return np.array(cim, dtype=np.int8)



def generate_IM(num_features, dimensionality):
    '''
        Initialises an ID Memory (IM). Each HV is randomly (orthogonally) initialised.
    '''
    return np.array(np.random.choice([-1,1], size=(num_features, dimensionality)), dtype=np.int8)



def encode_as_HVs_record_based(data, cim, im, quantile_midpoints):
    '''
        Encode a dataset as HVs

        The basic encoding procedure itself is very simple:
        for each sample:
            for each feature:
                select feature value HV, a, from CIM
                select feature ID HV, b, from IM
                compute bound feature pair HV as a * b (elementwise)
            bundle (elementwise add) all bound feature pairs from the sample, then sign result
    '''
    dimensionality = len(cim[0])

    num_features = len(im)
 
    data_HVs = np.empty((len(data), dimensionality), dtype=np.int16)

    #encoding by looping over features and doing indexing/additions considering all samples at once
    for f in range(num_features):
        indices = np.digitize(data[:,f], quantile_midpoints[f])
        data_HVs += np.multiply(im[f], np.take(cim, indices, axis=0))

    data_HVs = rand_sign(data_HVs.astype(np.float64))

    return data_HVs



def initialise_prototype_bundles(train_data_HVs, train_labels):
    '''
        Generates initial class prototype *bundles* given a set of input data and the generated CIMs (i.e. one-shot HDC training)
        TODO: finish this doc
    '''
    num_classes = len(np.unique(train_labels))
    dimensionality = train_data_HVs.shape[-1]

    class_bundles = np.zeros((num_classes, dimensionality))
    for i in range(len(train_data_HVs)):
        sample_HV = train_data_HVs[i]
        class_bundles[train_labels[i]] += sample_HV
    
    return class_bundles



def classify(sample_HV, class_prototypes):
    '''
        Classifies a single sample.
    '''
    sims = np.zeros(len(class_prototypes))
    for i in range(len(class_prototypes)):
        sims[i] = np.sum(np.multiply(sample_HV, class_prototypes[i]))

    return np.argmax(sims)



def evaluate_HDC(data, class_prototypes, labels):
    '''
        Computes the classification accuracy, given a set of samples (encoded as HVs), class prototypes, and labels.
    '''
    sims = np.dot(data.astype(np.int16), class_prototypes.T.astype(np.int16))
    preds = np.argmax(sims, axis=1)
    return np.mean(preds == labels)



def train_classifier_one_iteration(data_HVs, labels, class_bundles):
    '''
        Updates class prototypy bundles according to:
        for each sample HV, h:
            classify h
            if misclassified:
                add h to true class prototype bundle
                subtract h from falsely-predicted prototype bundle
    '''

    class_prototypes = rand_sign(class_bundles)

    #compute class predictions for each sample
    sims = np.dot(data_HVs.astype(np.int16), class_prototypes.T.astype(np.int16))         #int16 required to that output is int16 (and doesnt overflow)
    preds = np.argmax(sims, axis=1)

    #loop over each HV and update class prototype bundles if incorrectly classified
    for i in range(len(data_HVs)):
        pred = preds[i]
        if pred != labels[i]:
            class_bundles[pred] -= data_HVs[i]
            class_bundles[labels[i]] += data_HVs[i]

    return class_bundles







def train_encoder_one_iteration(train_data, train_labels, cim, im_bundles, quantile_midpoints, class_prototypes, shuffle_train_data=True, batch_size=50):
    #for incorrectly classified sample, update sample HV to be more similar to class prototype
    #do this by updating ID HVs

    dimensionality = len(cim[0])
    num_features = len(im_bundles)
        
    num_bits_to_update = int(np.floor(dimensionality/num_features))         #"non-overlapping" learning - each ID HV has D/f unique random components changed per update step

    updated_im_bundles = np.array(im_bundles)

    if batch_size is None:
        data_batches = [train_data]
        label_batches = [train_labels]
    else:
        if shuffle_train_data:
            p = np.random.permutation(len(train_labels))
            shuffled_train_data = train_data[p]
            shuffled_train_labels = train_labels[p]
            data_batches = [shuffled_train_data[i:i + batch_size] for i in range(0, len(shuffled_train_data), batch_size)]
            label_batches = [shuffled_train_labels[i:i + batch_size] for i in range(0, len(shuffled_train_labels), batch_size)]

        else:
            data_batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
            label_batches = [train_labels[i:i + batch_size] for i in range(0, len(train_labels), batch_size)]

    quant_class_prototypes = np.array(class_prototypes, np.int8)

    batch_im_updates_dict = {}
    batch_update_values_dict = {}
    #loops over each batch
    for batch_no in range(len(label_batches)):

        label_batch = label_batches[batch_no]
        data_batch = data_batches[batch_no]

        bipolar_im = np.array(rand_sign(updated_im_bundles), dtype=np.int8)

        sample_HVs = np.array(encode_as_HVs_record_based(data_batch, cim, bipolar_im, quantile_midpoints), np.int8)


        im_updates = np.zeros((num_features, dimensionality), np.int32)

        preds = [classify(sample_HVs[i], quant_class_prototypes) for i in range(len(data_batch))]

        batch_update_list = []
        #loops over each sample in the batch
        for i in range(len(data_batch)):
            #classify with classify function
            pred = preds[i]

            if pred != label_batch[i]: #update if misclassified
                current_raw_sample = data_batch[i]
                correct_prototype = quant_class_prototypes[label_batch[i]] #get the HV of the correct prototype
                incorrect_prototype = quant_class_prototypes[pred] #get the HV of the incorrect prototype
                prototype_diff = correct_prototype - incorrect_prototype #the update director

                #update item memory
                #for each level-ID binding, bind with the correct prototype, then bundle (a portion of) the result with the ID HV

                #indices used for masking - ensures each sample updates a random selection of bits in the IM bundle (each update selects different random bits)
                indices = np.arange(dimensionality)
                np.random.shuffle(indices)
                im_indices = indices[:num_features*num_bits_to_update].reshape([num_features, num_bits_to_update]) #what is this operation?

                #we have our vector, current_raw_sample, we want to construct the masked bound feature pairs of this sample, then multiply each one with the prototype diff, and add to the im_updates matrix at the correct position
                #1) select the CIM HVs that correspond to each of the values in current_raw_sample - get a list of level HVs, l
                #1a) take from this matrix only the indices that we are interested in
                #1b) take the indices of interest out of the IM as well
                #2) bind the level HVs with the corresponding ID HVS (matrix hadamard product) to get bound_feature_pairs
                #3) multiply the bound feature pairs with the prototype diff HV (filtered to only include indices of interest)
                #4) add the result to the updates bundle
                #4) once these updates have been aggregated for all samples in the batch, multiply by the IM, and add the result to the IM

                # idxs = np.digitize(current_raw_sample, quantile_midpoints)          #idxs is the array of discretised values of the raw feature vector, used to select level HVs from CIM
                idxs = np.sum(current_raw_sample[:, None] > quantile_midpoints, axis=1)     #np.digitize doesnt allow different bin edges for each element, so this is a workaround to enable it
                rows = np.arange(num_features)[:, None]     #rows of the IM, used for selecting only indices to update, along with im_indices
                l = np.take(cim, idxs, axis=0)[rows, im_indices] #how is rows different from im_indices? rows is used to select the correct level HV for each feature, while im_indices is used to select the indices of the IM to update
                bound_feature_pairs = np.multiply(l, bipolar_im[rows, im_indices])
                
                tiled_prototype_diff = np.tile(prototype_diff, (num_features,1))
                update = np.multiply(tiled_prototype_diff[rows, im_indices], bound_feature_pairs)

                batch_update_list.append(update)

                im_updates[rows, im_indices] += update

        im_updates = np.multiply(im_updates, bipolar_im)

        batch_im_updates_dict[f'batch_{batch_no}'] = im_updates
        batch_update_values_dict[f'batch_{batch_no}'] = batch_update_list #np.unique(im_updates[im_updates!=0])

        updated_im_bundles += im_updates

    # returns unbinarised item memory bundles
    return updated_im_bundles, im_updates, batch_update_list









##################################################################################################
#Commented functions are faster implementation of encoding. equivalent to above encoder.
##################################################################################################


# def init_encoder_worker(shm1_name, shape1, dtype1, shm2_name, shape2, dtype2):
#     '''
#         Initialisation func for parallelising encoding process.
#     '''

#     #not required, but ensures these subprocesses exit with default behaviour (prevents these subprocesses triggering the handle_sigterm process (used to allow for manual exiting in outer script)
#     signal.signal(signal.SIGTERM, signal.SIG_DFL)
#     signal.signal(signal.SIGINT, signal.SIG_DFL)

#     #making shared arrays (CIM and IM) globally accessible
#     global shared_array1, shared_array2, shm1, shm2

#     shm1 = shared_memory.SharedMemory(name=shm1_name)
#     shared_array1 = np.ndarray(shape1, dtype=dtype1, buffer=shm1.buf)

#     shm2 = shared_memory.SharedMemory(name=shm2_name)
#     shared_array2 = np.ndarray(shape2, dtype=dtype2, buffer=shm2.buf)


# def encode_HVs_batch(batch_args):
#     '''
#         Function for encoding a batch of samples as a parallel worker. Does not iterate over samples, but rather iterates over features for fewer iterations and more parallelism.
#         TODO: completely vectorise this
#     '''

#     batch, dimensionality, num_features, quantile_midpoints = batch_args
#     # print(f"[PID {os.getpid()}] Starting batch of size {len(batch)}")

#     data_HVs_batch = np.zeros((len(batch), dimensionality), dtype=np.int16)

#     for f in range(num_features):
#         indices = np.digitize(batch[:,f], quantile_midpoints[f])
#         data_HVs_batch += np.multiply(shared_array2[f], np.take(shared_array1, indices, axis=0))

#     data_HVs_batch = rand_sign(data_HVs_batch.astype(np.float64))
#     return data_HVs_batch



# def encode_as_HVs_record_based_batched(data, cim, im, quantile_midpoints, batch_size=200):
#     '''
#         encodes data in batches to speed up process (output is the same as encoder_as_HVs_record_based).
#     '''
#     dimensionality = len(cim[0])

#     num_features = len(im)

#     #if batch size is bigger than the size of the data (i.e. if data already batched by training procedure) or batch_encode explicitly turned off, then treat as normal (i.e. dont batch encode)
#     if len(data) <= batch_size:
#         data_batches = [data]
#         batch_size = len(data)
#     else:
#         data_batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

#     #copied to each process: dimensionality, num_features, quantile_midpoints
#     #shared between workers: im, cim

#     #creating shared memory for CIM and IM, and copying them in
#     cim_shm = shared_memory.SharedMemory(create=True, size=cim.nbytes)
#     np.ndarray(cim.shape, dtype=cim.dtype, buffer=cim_shm.buf)[:] = cim
#     im_shm = shared_memory.SharedMemory(create=True, size=im.nbytes)
#     np.ndarray(im.shape, dtype=im.dtype, buffer=im_shm.buf)[:] = im

#     data_batch_args = [(i, dimensionality, num_features, quantile_midpoints) for i in data_batches]
    
#     try:
#         with Pool(
#             processes=6,
#             initializer=init_encoder_worker,
#             initargs=(
#                 cim_shm.name, cim.shape, cim.dtype, im_shm.name, im.shape, im.dtype
#             )
#         ) as pool:
#             data_HVs = pool.map(encode_HVs_batch, data_batch_args, chunksize=1)

#         data_HVs = np.vstack(data_HVs, dtype=np.int8)
#     finally:
#         cim_shm.close()
#         cim_shm.unlink()
#         im_shm.close()
#         im_shm.unlink()


#     return data_HVs