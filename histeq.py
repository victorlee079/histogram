import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt


def main(argv):
    input_img = "sample.jpg"
    output_img = "output.jpg"

    if len(argv) > 1:
        input_img = argv[1]
    if len(argv) > 2:
        output_img = argv[2]

    hsv_img = read_to_hsv(input_img)
    (H, S, V) = cv2.split(hsv_img)
    new_v = equalize(V, 256)
    write_equalized_img(output_img, (H, S, new_v))


def write_equalized_img(fname, mv):
    equalized_img = cv2.merge(mv)
    cv2.imwrite(fname, cv2.cvtColor(equalized_img, cv2.COLOR_HSV2BGR))


def equalize(V, L):
    (M, N) = V.shape

    vcount = np.zeros(L, dtype=int)
    mapping = np.zeros(L, dtype=int)
    vlist = V.flatten()

    gen_hist(vlist, L, "Original Histogram", "original_histogram")

    print("Equalizing the images")
    ulist, ucount = np.unique(vlist, return_counts=True)
    for i in range(len(ulist)):
        vcount[ulist[i]] = ucount[i]
    for i in range(len(mapping)):
        mapping[i] = round((L - 1) / (M * N) * np.sum(vcount[:i]))
    for i in range(len(vlist)):œ
        vlist[i] = mapping[vlist[i]]

    gen_hist(vlist, L, "Equalized Histogram", "equalized_histogram")

    new_v = np.reshape(vlist, (M, N))
    return new_v


def gen_hist(data, bsize, title, fname):
    print("Generating " + fname + ".png")
    plt.hist(data, bins=bsize)
    plt.title(title)
    plt.savefig(fname + ".png")
    plt.clf()


def read_to_hsv(img_file):
    print("Reading file: ", img_file)
    img = cv2.imread(img_file)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv_img


if __name__ == "__main__":
    main(sys.argv)
