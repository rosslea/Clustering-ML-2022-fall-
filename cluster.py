import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
import pickle

def clustering(data):
    sets = {}
    Z = linkage(data)
    f = fcluster(Z, t=0.5)
    max_set_id = max(f)
    cls = np.array(f)
    for i in range(max_set_id):
        set_id = i + 1
        idx = np.nonzero(cls == set_id)
        if len(list(*idx)) != 1:
            sets[set_id] = list(*idx)
    return sets

def main():
    data = np.load('./data/d_mat.npy')
    sets = clustering(data)
    with open('./data/clusters.pkl', 'wb') as f:
        pickle.dump(sets, f)

if __name__ == '__main__':
    main()