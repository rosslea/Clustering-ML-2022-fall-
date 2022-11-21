import cv2
from pathlib import Path
def main():
    lst = [f'./data/output/out_{i:05d}0.jpg' for i in range(541)]
    video = cv2.VideoWriter('./data/video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),2,(720,576))

    for file in lst:
        img = cv2.imread(str(file))
        img = cv2.resize(img,(720,576))
        video.write(img)

    video.release()

if __name__ == '__main__':
    main()