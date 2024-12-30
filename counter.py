# -*- coding: utf-8 -*-

# import libraries
from skimage import io, measure, filters, segmentation, morphology, color, exposure
from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max
from skimage.segmentation import watershed

from math import sqrt
import numpy as np
import pandas as pd

from scipy import ndimage

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['image.cmap'] = 'inferno'

from plotting_functions import (plot_bboxs, plot_texts, plot_circles,
                                 easy_sub_plot)

from image_processing_functions import (invert_image, crop_circle,
                                         background_subtraction,
                                         search_for_blobs,
                                         make_circle_label,
                                         detect_circle_by_canny)

class Counter():
    def __init__(self, image_path=None, image_array=None, verbose=False):
        self._props = []
        self.props = {}
        self.verbose = verbose

        if not image_path is None:
            self.load_from_path(image_path, verbose=verbose)
        elif not image_array is None:
            self.load_image(image_array, verbose=verbose)

    def load_from_path(self, image_path, verbose=True):
        image = io.imread(image_path)
        self.load_image(image, verbose=verbose)

    def load_image(self, image_array, verbose=True):
        """
        Load and preprocess the image, handling both RGB and grayscale formats
        """
        try:
            # Convert to numpy array if not already
            self.image_raw = np.array(image_array)
            
            # Convert to RGB if grayscale
            if len(self.image_raw.shape) == 2:
                self.image_raw = np.dstack([self.image_raw] * 3)
            elif len(self.image_raw.shape) == 3:
                if self.image_raw.shape[2] == 4:  # RGBA image
                    self.image_raw = self.image_raw[:, :, :3]
                elif self.image_raw.shape[2] == 1:
                    self.image_raw = np.dstack([self.image_raw[:, :, 0]] * 3)
            
            # Convert to grayscale for processing
            self.image_bw = color.rgb2gray(self.image_raw)
            self.image_inverted_bw = invert_image(self.image_bw)
            
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

    def detect_area_by_canny(self, n_samples=None, radius=300, n_peaks=20, verbose=True):
        """
        Detect areas in the image using Canny edge detection
        """
        try:
            if verbose:
                print("Detecting sample area...")

            # Segmentation
            bw = self.image_bw.copy()
            labeled = detect_circle_by_canny(bw, radius=radius, n_peaks=n_peaks)
            
            if labeled is None:
                raise ValueError("No circles detected in the image")
            
            self.labeled = labeled

            # Region properties
            props = measure.regionprops(label_image=labeled, intensity_image=self.image_bw)
            if not props:
                raise ValueError("No regions found in the image")
                
            props = np.array(props)
            bboxs = np.array([prop.bbox for prop in props])
            areas = np.array([prop.area for prop in props])
            cordinates = np.array([prop.centroid for prop in props])
            eccentricities = np.array([prop.eccentricity for prop in props])

            # Filter objects
            area_threshold = np.percentile(areas, 90)
            selected = (areas >= area_threshold) & (eccentricities < 0.3)
            
            if not np.any(selected):
                selected = areas == areas.max()

            # Update labels and get final properties
            labeled = make_circle_label(bb_list=bboxs[selected], img_shape=self.image_bw.shape)
            props = measure.regionprops(label_image=labeled, intensity_image=self.image_bw)
            
            if not props:
                raise ValueError("No regions found after filtering")
                
            props = np.array(props)
            bboxs = np.array([prop.bbox for prop in props])
            areas = np.array([prop.area for prop in props])
            cordinates = np.array([prop.centroid for prop in props])
            eccentricities = np.array([prop.eccentricity for prop in props])

            # Sort by y-coordinate
            idx = np.argsort(cordinates[:, 0])
            self._props = props[idx]
            self.props["bboxs"] = bboxs[idx]
            self.props["areas"] = areas[idx]
            self.props["cordinates"] = cordinates[idx]
            self.props["eccentricities"] = eccentricities[idx]
            self.props["names"] = [f"sample_{i}" for i in range(len(areas))]

            if verbose:
                print(f"Detected {len(areas)} samples")

        except Exception as e:
            print(f"Error in detect_area_by_canny: {str(e)}")
            # Fallback to using entire image as one sample
            self._props = [measure.regionprops(np.ones_like(self.image_bw, dtype=int), 
                                             intensity_image=self.image_bw)[0]]
            self.props["bboxs"] = np.array([[0, 0, self.image_bw.shape[0], self.image_bw.shape[1]]])
            self.props["areas"] = np.array([self.image_bw.size])
            self.props["cordinates"] = np.array([[self.image_bw.shape[0]/2, self.image_bw.shape[1]/2]])
            self.props["eccentricities"] = np.array([0.0])
            self.props["names"] = ["sample_0"]
            
            if verbose:
                print("Using entire image as one sample")

    def plot_detected_colonies(self, ax=None, plot="final", vmax=None, overlay_circle=True):
        """
        Plot detected colonies on the given axes
        
        Args:
            ax (matplotlib.axes.Axes): Axes to plot on
            plot (str): Type of plot - "raw", "final", or "raw_inversed"
            vmax (float): Maximum value for color scaling
            overlay_circle (bool): Whether to overlay detected circles
        """
        if plot == "raw":
            image = self.sample_image_bw[0]
        elif plot == "final":
            image = self.sample_image_processed[0]
        elif plot == "raw_inversed":
            image = self.sample_image_inversed_bw[0]
        else:
            raise ValueError("Invalid plot type")

        if vmax is None:
            vmax = _get_vmax([image])

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if plot == "raw":
            ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
        else:
            ax.imshow(image, vmin=0, vmax=vmax)

        if overlay_circle and hasattr(self, 'detected_blobs'):
            blobs = self.detected_blobs[0]
            plot_circles(blobs, ax=ax)
            ax.set_title(f"{self.props['names'][0]}: {len(blobs)} colonies")
        
        return ax

    def crop_samples(self, shrinkage_ratio=0.9):
        self.sample_image_bw = [crop_circle(i.intensity_image, shrinkage_ratio) for i in self._props]
        self.sample_image_inversed_bw = [crop_circle(invert_image(i.intensity_image), shrinkage_ratio) for i in self._props]
        self.sample_image_processed = self.sample_image_inversed_bw.copy()

    def subtract_background(self, sigma=1, verbose=True, reset_image=True):
        if reset_image:
            self.sample_image_processed = self.sample_image_inversed_bw.copy()
            
        for i, image in enumerate(self.sample_image_processed):
            result = background_subtraction(image=image, sigma=sigma, verbose=False)
            result = result - result[0,0]
            result[result<0] = 0
            self.sample_image_processed[i] = result

    def detect_colonies(self, min_size=5, max_size=15, threshold=0.02, num_sigma=10, overlap=0.5, verbose=True):
        self.detected_blobs = []
        for image in self.sample_image_processed:
            blobs = search_for_blobs(image=image, min_size=min_size, max_size=max_size, 
                                   num_sigma=num_sigma, overlap=overlap,
                                   threshold=threshold, verbose=False)
            self.detected_blobs.append(blobs)

        if verbose:
            print(f"Detected {len(self.detected_blobs[0])} colonies in sample")

    def plot_detected_area(self):
        print(str(len(self.props['areas'])) +" samples were detected")
        ax = plt.axes()
        plt.title("detected samples")
        ax.imshow(self.image_raw)
        plot_bboxs(bbox_list=self.props["bboxs"], ax=ax)
        plot_texts(text_list=self.props["names"], cordinate_list=self.props["bboxs"], ax=ax, shift=[0, -60])

    def adjust_contrast(self, verbose=True, reset_image=False):
        if reset_image:
            self.sample_image_processed = self.sample_image_inversed_bw.copy()
        if verbose:
            print("before_contrast_adjustment")
            vmax = _get_vmax(self.sample_image_processed)
            easy_sub_plot(self.sample_image_processed, 4, self.props["names"], {"vmin":0, "vmax": vmax})

        for i, image in enumerate(self.sample_image_processed):
            result = exposure.adjust_log(image, 1)
            #result = result - result[0,0]
            self.sample_image_processed[i] = result

        if verbose:
            print("after_contrast_adjustment")
            vmax = _get_vmax(self.sample_image_processed)
            easy_sub_plot(self.sample_image_processed, 4, self.props["names"], {"vmin":0, "vmax": vmax})

    def plot_cropped_samples(self, inverse=False, col_num=3):
        if not inverse:
            image_list = self.sample_image_bw
            vmax = _get_vmax(image_list)
            easy_sub_plot(image_list, col_num, self.props["names"], args={"cmap": "gray", "vmin": 0, "vmax": vmax})

        if inverse:
            image_list = self.sample_image_inversed_bw
            vmax = _get_vmax(image_list)
            easy_sub_plot(image_list, col_num, self.props["names"], args={"vmin": 0, "vmax": vmax})

def _get_vmax(image_list):
    vmax = []
    for i in image_list:
        vmax.append(i.max())
    vmax = np.max(vmax)
    return vmax