import matplotlib.pyplot as plt
import dask.array as da
import time
from scipy.signal import convolve2d
import psutil
import numpy as np
from PIL import Image
import os
import glob

# Without Dask
def grayscale(image):
    img_array = np.array(image)
    grayscale_image = img_array.mean(axis=2, dtype='uint8')
    return Image.fromarray(grayscale_image)

def grayscale_dask(image):
    img_array = np.array(image)
    dask_image = da.from_array(img_array, chunks=(100, 100, 3))
    grayscale_image = dask_image.mean(axis=2, dtype='uint8').compute()
    return Image.fromarray(grayscale_image)

def gaussian_blur(image, sigma=1):
    img_array = np.array(image)
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) *
                                       np.exp(-((x-(kernel_size-1)/2)**2 +
                                                (y-(kernel_size-1)/2)**2)/(2*sigma**2)),
                             (kernel_size, kernel_size))
    kernel /= np.sum(kernel)
    blurred_image = np.zeros_like(img_array, dtype=np.float32)
    for c in range(3):
        blurred_image[:, :, c] = convolve2d(img_array[:, :, c], kernel, mode='same')
    blurred_image = Image.fromarray(np.clip(blurred_image, 0, 255).astype(np.uint8))
    return blurred_image

def gaussian_blur_dask(image, sigma=1):
    img_array = np.array(image)
    kernel_size = int(6 * sigma + 1)
    if (kernel_size % 2) == 0:
        kernel_size += 1
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) *
                                       np.exp(-((x-(kernel_size-1)/2)**2 +
                                                (y-(kernel_size-1)/2)**2)/(2*sigma**2)),
                             (kernel_size, kernel_size))
    kernel /= np.sum(kernel)
    dask_image = da.from_array(img_array, chunks=(100, 100, 3))
    def convolve_channel(channel, kernel):
        return convolve2d(channel, kernel, mode='same')
    blurred_image = dask_image.map_blocks(lambda block: np.stack([convolve_channel(block[:, :, c], kernel) for c in range(3)], axis=-1), dtype=np.float32)
    blurred_image = blurred_image.compute()
    blurred_image = Image.fromarray(np.clip(blurred_image, 0, 255).astype(np.uint8))
    return blurred_image

def measure_performance(func, *args, **kwargs):
    process = psutil.Process()
    start_time = time.time()
    cpu_times_start = process.cpu_times()

    # Measure memory before the function call
    memory_start = process.memory_info().rss

    latency_start = time.time()
    result = func(*args, **kwargs)
    latency_end = time.time()

    end_time = time.time()
    cpu_times_end = process.cpu_times()

    # Measure memory after the function call
    memory_end = process.memory_info().rss

    elapsed_time = end_time - start_time
    user_cpu_time = cpu_times_end.user - cpu_times_start.user
    system_cpu_time = cpu_times_end.system - cpu_times_start.system
    cpu_time = user_cpu_time + system_cpu_time

    # Calculate memory used
    if isinstance(result, da.Array):
        memory_used = result.nbytes
    else:
        memory_used = memory_end - memory_start

    latency = latency_end - latency_start

    return result, elapsed_time, cpu_time, memory_used, latency


def process_images(folder_path):
    if not os.path.exists(folder_path):
        print(f"The folder path {folder_path} does not exist.")
        return {}, {}, {}, {}

    print(f"Processing images in folder: {folder_path}")

    jpg_image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    jpeg_image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.jpeg')]
    image_paths = jpg_image_paths + jpeg_image_paths

    print(f"Number of images found: {len(image_paths)}")

    if not image_paths:
        print("No images found in the specified folder.")
        return {}, {}, {}, {}

    results = {
        "grayscale": {"results": [], "total_time": 0, "total_memory": 0},
        "grayscale_dask": {"results": [], "total_time": 0, "total_memory": 0},
        "blur": {"results": [], "total_time": 0, "total_memory": 0},
        "blur_dask": {"results": [], "total_time": 0, "total_memory": 0},
    }

    for image_path in image_paths:
        try:
            pil_image = Image.open(image_path)
            print(f"Processing {image_path}...")

            # Measure performance for grayscale without Dask
            _, elapsed_time, cpu_time, memory_used, latency = measure_performance(grayscale, pil_image)
            throughput = 1 / elapsed_time
            results["grayscale"]["results"].append((elapsed_time, cpu_time, memory_used, latency, throughput))
            results["grayscale"]["total_time"] += elapsed_time
            results["grayscale"]["total_memory"] += memory_used

            # Measure performance for grayscale with Dask
            _, elapsed_time, cpu_time, memory_used, latency = measure_performance(grayscale_dask, pil_image)
            throughput = 1 / elapsed_time
            results["grayscale_dask"]["results"].append((elapsed_time, cpu_time, memory_used, latency, throughput))
            results["grayscale_dask"]["total_time"] += elapsed_time
            results["grayscale_dask"]["total_memory"] += memory_used

            # Measure performance for Gaussian blur without Dask
            _, elapsed_time, cpu_time, memory_used, latency = measure_performance(gaussian_blur, pil_image, sigma=2)
            throughput = 1 / elapsed_time
            results["blur"]["results"].append((elapsed_time, cpu_time, memory_used, latency, throughput))
            results["blur"]["total_time"] += elapsed_time
            results["blur"]["total_memory"] += memory_used

            # Measure performance for Gaussian blur with Dask
            _, elapsed_time, cpu_time, memory_used, latency = measure_performance(gaussian_blur_dask, pil_image, sigma=2)
            throughput = 1 / elapsed_time
            results["blur_dask"]["results"].append((elapsed_time, cpu_time, memory_used, latency, throughput))
            results["blur_dask"]["total_time"] += elapsed_time
            results["blur_dask"]["total_memory"] += memory_used

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return results

def print_performance_results(results, description):
    if not results["results"]:
        print(f"No results for {description}.")
        return

    elapsed_times, cpu_times, memory_usages, latencies, throughputs = zip(*results["results"])
    avg_elapsed_time = np.mean(elapsed_times)
    avg_cpu_time = np.mean(cpu_times)
    avg_memory_usage = np.mean(memory_usages)
    avg_latency = np.mean(latencies)
    avg_throughput = np.mean(throughputs)

    print(f"{description}:")
    print(f"  Total elapsed time: {results['total_time']:.4f} seconds")
    print(f"  Total memory used: {results['total_memory'] / 1e6:.2f} MB")
    print(f"  Average elapsed time: {avg_elapsed_time:.4f} seconds")
    print(f"  Average CPU time: {avg_cpu_time:.4f} seconds")
    print(f"  Average memory usage: {avg_memory_usage / 1e6:.2f} MB")
    print(f"  Average latency: {avg_latency:.4f} seconds")
    print(f"  Average throughput: {avg_throughput:.2f} images/sec")

# Folder containing the images
folder_path = 'D:/DSE2/poze'

results = process_images(folder_path)

print_performance_results(results["grayscale"], "Grayscale without Dask")
print_performance_results(results["grayscale_dask"], "Grayscale with Dask")
print_performance_results(results["blur"], "Gaussian blur without Dask")
print_performance_results(results["blur_dask"], "Gaussian blur with Dask")
