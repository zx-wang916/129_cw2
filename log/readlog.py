import numpy as np


def read_log(filename):
    with open(filename, 'r', encoding='utf8') as f:
        dice = []
        pa = []
        iou = []
        for l in f.readlines():
            if not l.startswith('epoch'):
                continue

            l = l.split('|')

            if l[1].strip() == 'train':
                continue

            dice.append(float(l[2].strip().split(':')[1].strip()))
            pa.append(float(l[3].strip().split(':')[1].strip()))
            iou.append(float(l[4].strip().split(':')[1].strip()))

        dice_max_idx = np.argmax(dice)
        pa_max_idx = np.argmax(pa)
        iou_max_idx = np.argmax(iou)

        print(filename)
        print(dice_max_idx, dice[dice_max_idx], pa[dice_max_idx], iou[dice_max_idx])
        print(pa_max_idx, dice[pa_max_idx], pa[pa_max_idx], iou[pa_max_idx])
        print(iou_max_idx, dice[iou_max_idx], pa[iou_max_idx], iou[iou_max_idx])


def clear_progress(filename):
    with open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            if line.startswith('epoch'):
                print(line)


if __name__ == '__main__':
    read_log('sup.txt')
    read_log('semi.txt')
    read_log('cla.txt')

    # clear_progress('sup.txt')
    # clear_progress('semi.txt')
