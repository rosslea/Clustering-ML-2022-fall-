import numpy as np
import pickle
from tqdm import tqdm
def make_gauss(sigma):
    def gauss_func(x, y):
        index = -(x**2 + y**2)/(2*sigma**2)
        return 1 / (2 * np.pi * sigma**2) * np.exp(index)
    return gauss_func

gauss_1 = make_gauss(sigma=0.5)
gauss_2 = make_gauss(sigma=1.2)
gauss_3 = make_gauss(sigma=3.7)
gauss_4 = make_gauss(sigma=7.6)

def get_ids(data):
    ids = []
    for time in range(len(data)):
        max_id = max(list(data[time].keys()))
        ids.append(max_id)
    return max(ids)

def get_dph(data, idx):
    row, col = idx
    ks = data.keys()
    x, y = data.get(row, -1), data.get(col, -1)
    if x != -1 and y != -1:
        x, y = x[0] - y[0], x[1] - y[1]
        dis = gauss_1(x, y) + gauss_2(x, y) + gauss_3(x, y) + gauss_4(x, y)
        return dis/4
    else:
        return -1

def get_distance_one_frame(data, max_id):
    mat = np.zeros((max_id, max_id))

    for i in range(max_id):
        for j in range(i, max_id):
            idx = (i, j) # row col
            mat[i][j] = get_dph(data, idx)
            mat[j][i] = mat[i][j]
    return mat

def get_distance_all(data):
    max_id = get_ids(data)
    d_mat = []
    for time in tqdm(range(len(data))):
        d_mat.append(get_distance_one_frame(data[time], max_id))
    return np.stack(d_mat, axis=2)

def average_time(mat):
    d_mat = mat.sum(axis=-1)/mat.shape[-1]
    return d_mat

def main():
    with open('./data/ordered_data.pkl', 'rb') as f:
        data = pickle.load(f)
    # data = data[:2]
    mat = get_distance_all(data)
    d_mat = average_time(mat)
    np.save('./data/d_mat.npy', d_mat)

if __name__ == '__main__':
    main()