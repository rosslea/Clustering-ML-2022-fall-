from pathlib import Path
import pickle

def load():
    with open('./data/TrajectoryData_students003/students003.txt', 'r') as f:
        data = f.readlines()
    data  = [d[:-1].split('\t') for d in data]
    return data

def time_order(data):
    time_set = set([int(float(i[0])) for i in data])
    time_len = len(time_set)
    reorder = [{} for _ in range(time_len)]
    for d in data:
        time_index = int(float(d[0]))//10
        id = int(float(d[1]))
        reorder[time_index][id]=(float(d[2]), float(d[3]))
    return reorder

def main():
    data = load()
    data_order = time_order(data)
    with open('./data/ordered_data.pkl', 'wb') as f:
        pickle.dump(data_order, f)
    return 0

if __name__ == '__main__':
    main()




