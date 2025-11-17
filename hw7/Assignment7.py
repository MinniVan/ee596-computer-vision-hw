import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import convolve2d 
import os

# region DEBUG
DEBUG = False


def print_debug(*args, **kwargs):
    if DEBUG:
        print(args, kwargs)


if __name__ == "__main__":
    DEBUG = os.environ.get("PYTHON_DEBUG_MODE")
    if DEBUG is not None and DEBUG.lower() == "true":
        DEBUG = True
        print("DEBUG mode is enabled")
# endregion

MAX_DISPARITY = 30
SMOOTH_KERNEL_SIZE = 5


def load_image_in_grayscale(filepath) -> np.ndarray:
    return cv.imread(filepath, cv.IMREAD_GRAYSCALE)


def sum_of_abs_diff(nparray1: np.array, nparray2: np.array) -> int:
    return (np.abs(nparray1 - nparray2)).sum().item()


def scanlines(tb_left: np.array, tb_right: np.array) -> int:
    row_idx = 152
    col_idx1 = 102
    col_len = 100  # columns 102..201 inclusive

    tb_left_cropped = tb_left[row_idx, col_idx1 : col_idx1 + col_len]

    g_best = None
    d_best = None

    # we cap disparity by MAX_DISPARITY and also stay within bounds
    for d in range(MAX_DISPARITY + 1):
        start = col_idx1 - d
        end = start + col_len
        if start < 0 or end > tb_right.shape[1]:
            continue
        tb_right_cropped = tb_right[row_idx, start:end]
        g = sum_of_abs_diff(tb_left_cropped, tb_right_cropped)
        if g_best is None or g < g_best:
            g_best, d_best = g, d
    return d_best



def plot_1d_array(array, title, xlabel=None, ylabel=None, save_image=True):
    domain = range(len(array))
    plt.plot(domain, array, marker="o")
    plt.xlabel(title)
    plt.ylabel(xlabel)
    plt.title(ylabel)
    plt.grid(True)
    if save_image:
        plt.savefig(f"figure/{title}.png")
    plt.show()


def plot_2d_array_as_image(array2d: np.array, title, save_image=True):
    plt.imshow(array2d, cmap="gray")
    plt.title(title)
    plt.colorbar()
    if save_image:
        os.makedirs("figure", exist_ok=True)
        plt.savefig(f"figure/{title}.png")
    plt.show()


def shift_array(nparray: np.array, d: int) -> np.array:
    shifted = np.zeros_like(nparray)
    if d == 0:
        shifted[:, :] = nparray[:, :]
    elif d > 0:
        shifted[:, d:] = nparray[:, :-d]
    elif d < 0:
        shifted[:, : nparray.shape[1] + d] = nparray[:, -d:]
    return shifted


if DEBUG:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (shift_array(a, 1) == [[0, 1, 2], [0, 4, 5], [0, 7, 8]]).all()
    assert (shift_array(a, 2) == [[0, 0, 1], [0, 0, 4], [0, 0, 7]]).all()


def auto_correlation(tb_right):
    auto_correlations = []
    for d in range(MAX_DISPARITY + 1):
        abs_diff_image = np.abs(tb_right - shift_array(tb_right, d))
        auto_correlations.append(abs_diff_image[152, 152])

    if DEBUG:
        plot_1d_array(
            auto_correlations,
            title="auto_correlation",
            xlabel="disparity d",
            ylabel="|R - R_shifted(d)| at (152,152)",
        )
    return auto_correlations


def convolve2d_box(array: np.array, kernel_size: int) -> np.array:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    convolved = convolve2d(array, kernel, mode="same", boundary="fill", fillvalue=0)
    return convolved


def smoothing(tb_right):
    smoothed_auto = []
    for d in range(MAX_DISPARITY + 1):
        abs_diff_image = np.abs(tb_right - shift_array(tb_right, d))
        smoothed_image = convolve2d_box(abs_diff_image, SMOOTH_KERNEL_SIZE)
        smoothed_auto.append(smoothed_image[152, 152])

    if DEBUG:
        plot_1d_array(
            smoothed_auto,
            title="smoothed_auto_correlation",
            xlabel="disparity d",
            ylabel="smoothed |R - R_shifted(d)| at (152,152)",
        )
    return smoothed_auto


def cross_correlation(tb_left, tb_right):
    cross_corr = []
    for d in range(MAX_DISPARITY + 1):
        shifted_right = shift_array(tb_right, d)
        abs_diff_image = np.abs(tb_left - shifted_right)
        cross_corr.append(abs_diff_image[152, 152])

    if DEBUG:
        plot_1d_array(
            cross_corr,
            title="cross_correlation",
            xlabel="disparity d",
            ylabel="|L - R_shifted(d)| at (152,152)",
        )
    return cross_corr


def disparity_map(
    tb_left: np.ndarray,
    tb_right: np.ndarray,
    max_disparity: int = MAX_DISPARITY,
    kernel_size: int = SMOOTH_KERNEL_SIZE,
    plot_result: bool = False,
) -> np.ndarray:
    h, w = tb_left.shape
    cost_volume = np.zeros((h, w, max_disparity + 1), dtype=np.float32)

    for d in range(max_disparity + 1):
        shifted_right = shift_array(tb_right, d)
        abs_diff = np.abs(tb_left - shifted_right).astype(np.float32)
        smoothed = convolve2d_box(abs_diff, kernel_size)
        cost_volume[:, :, d] = smoothed

    # argmin over disparities → disparity map (non-negative disparities)
    disparity = np.argmin(cost_volume, axis=2).astype(np.int32)

    if plot_result or DEBUG:
        plot_2d_array_as_image(disparity, "disparity_left_right")

    return disparity


def right_left_disparity(
    tb_left: np.ndarray,
    tb_right: np.ndarray,
    max_disparity: int = MAX_DISPARITY,
    kernel_size: int = SMOOTH_KERNEL_SIZE,
    plot_result: bool = False,
) -> np.ndarray:
    h, w = tb_right.shape
    cost_volume = np.zeros((h, w, max_disparity + 1), dtype=np.float32)

    for d in range(max_disparity + 1):
        shifted_left = shift_array(tb_left, -d)  # shift LEFT by d
        abs_diff = np.abs(tb_right - shifted_left).astype(np.float32)
        smoothed = convolve2d_box(abs_diff, kernel_size)
        cost_volume[:, :, d] = smoothed

    disp_indices = np.argmin(cost_volume, axis=2).astype(np.int32)
    # right-left disparities are negative
    #disparity_rl = -disp_indices
    disparity_rl = disp_indices

    if plot_result or DEBUG:
        plot_2d_array_as_image(disparity_rl, "disparity_right_left")

    return disparity_rl


def disparity_check(
    tb_left: np.ndarray,
    tb_right: np.ndarray,
    max_disparity: int = MAX_DISPARITY,
    kernel_size: int = SMOOTH_KERNEL_SIZE,
    plot_result: bool = True,
) -> np.ndarray:
    dL = disparity_map(tb_left, tb_right,
                       max_disparity, kernel_size, plot_result=False)
    dR = right_left_disparity(tb_left, tb_right,
                              max_disparity, kernel_size, plot_result=False)
    h, w = dL.shape
    cleaned = np.zeros_like(dL, dtype=np.int32)
    mask = (dL > 0) & (dL == dR)
    cleaned[mask] = dL[mask]


    #if plot_result or DEBUG:
    plot_2d_array_as_image(cleaned, "disparity_cleaned")

    return cleaned

def reconstruction(
    tb_left: np.ndarray,
    tb_right: np.ndarray,
    max_disparity: int = MAX_DISPARITY,
    kernel_size: int = SMOOTH_KERNEL_SIZE,
    ply_filename: str = "kermit.ply",
) -> str:
    # Fill your code hear
    # get cleaned disparity map
    cleaned_disparity = disparity_check(tb_left, tb_right,
                                        max_disparity, kernel_size, plot_result=False)
    # load original left color imgage for RGB values
    tb_left_color = cv.imread("tsukuba_left.png", cv.IMREAD_COLOR)
    if tb_left_color is None:
        raise FileNotFoundError("Could not load tsukuba_left.png in color")
    h, w = cleaned_disparity.shape

    #nomial camera vaues
    f = 1.0  # focal length
    B = 1.0  # baseline

    points_3d = []

    for y in range(h):
        for x in range(w):
            d = cleaned_disparity[y, x]
            if d <= 0:
                continue  # skip invalid disparities
            # center pixel cooords
            X = x - w / 2.0
            Y = y - h / 2.0
            Z = (f * B) / float(d)

            # clor from left image( open CV uses BGR format)
            b, g, r = tb_left_color[y, x]
            points_3d.append((X, Y, Z, int(r), int(g), int(b)))

    # write to PLY file
    with open(ply_filename, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(points_3d)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        for X, Y, Z, r, g, b in points_3d:
            ply_file.write(f"{X} {Y} {Z} {r} {g} {b}\n")
    print_debug(f"Saved PLY with {len(points_3d)} points to {ply_filename}")
    return ply_filename

if __name__ == "__main__":
    tb_left = load_image_in_grayscale("tsukuba_left.png")
    tb_right = load_image_in_grayscale("tsukuba_right.png")
    # scanlines(tb_left, tb_right)
    # auto_correlation(tb_right)
    # smoothing(tb_right)
    # cross_correlation(tb_left, tb_right)
    # disparity_map(tb_left, tb_right, plot_result=True)
    # right_left_disparity(tb_left, tb_right, plot_result=True)
    disparity_check(tb_left, tb_right)
    reconstruction(tb_left, tb_right)
    
