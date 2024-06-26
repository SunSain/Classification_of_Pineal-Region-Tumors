import numpy as np
import xrange

def compute_knn(batch_feat, trainLabels, knn=5,  epoch = 8):
    ''' compute the knn according to instance id/ class id
    '''
    dist_feat  = np.matmul(batch_feat, batch_feat.T)
    #nn_index = compute_knn(dist_feat, trainLabels, knn=1, epoch = epoch)
    dist_feat
    ndata = len(trainLabels)   
    nnIndex = np.arange(ndata)
    
    top_acc = 0.
    # compute the instance knn
    for i in xrange(ndata):
        dist_feat[i,i] = -1000
        dist_tmp = dist_feat[i,:]
        ind = np.argpartition(dist_tmp, -knn)[-knn:]
        # random 1nn and augmented sample for positive 
        nnIndex[i] = np.random.choice([ind[0],i])
    return nnIndex.astype(np.int32)