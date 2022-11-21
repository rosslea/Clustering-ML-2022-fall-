import pickle
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import skimage.io as io
from tqdm import tqdm

def cm2pixel(pos, dpi=125):
    x, y = pos
    x, y = x / 2.54 * dpi, y / 2.54 * dpi
    y = 575 - y
    return (x,y)

def check_valid(dict_value, data):
    ids = [i for i in data.keys()]
    for i in dict_value:
        if i not in ids:
            return False
    return True


def vis_all():
    with open('./data/ordered_data.pkl', 'rb') as f:
        pos = pickle.load(f)

    # load set
    with open('./data/clusters.pkl', 'rb') as f:
        data = pickle.load(f)
    set_len = len(data.keys())
    
    # set color for each cluster
    all_colors = np.linspace(0, 100, 10)

    # iter over imgs
    for i in tqdm(range(541)):
    # for i in range(1):
        name = f'./data/images/{i:05d}0.jpg'
        img = io.imread(name)

        my_dpi = 96
        w = 720
        h = 576
        fig, ax = plt.subplots(figsize=(w / my_dpi, h / my_dpi), dpi=my_dpi, constrained_layout=True)
        ax.axis('off')
        ax.imshow(img)

        # loop over clusters
        for idx, (k, v) in enumerate(data.items()):
            patches = []
            # circle this cluster
            if check_valid(v, pos[i]):
                for pt in list(v):
                    p = pos[i][pt]
                    x, y = cm2pixel(p, 122)
                    if x < 0 or y < 0:
                        continue
                    wd = 45
                    ht = 45
                    if x + wd > 720 or y + ht > 576:
                        continue
                    x, y = x - wd//2, y - ht//2
                    fancybox = mpatches.FancyBboxPatch((x, y), wd, ht, boxstyle=mpatches.BoxStyle("Round", pad=0.02))
                    patches.append(fancybox)
            else:
                continue
            
            colors = np.array([all_colors[k%10]]*len(patches))
            # colors = np.array([0.3]*len(patches))
            # colors = 100*np.random.rand(len(patches))
            # print(colors)
            collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.5)
            collection.set_array(colors)
            collection.set_clim([5,150])
            ax.add_collection(collection)
                # print('cluster', idx)
        
        plt.margins(0, 0)
        plt.savefig(f'./data/output/out_{i:05d}0.jpg', bbox_inches='tight', pad_inches=0.0)
        plt.close()




if __name__ == '__main__':
    vis_all()