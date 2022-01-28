import numpy as np
from numpy.typing import *

class Conv3x3:
    """
    https://victorzhou.com/blog/intro-to-cnns-part-1/

    3x3 尺寸过滤器卷积层

    图片尺寸：HxW
    填充后尺寸：(H+2)x(W+2)
      - 原图四周增加一个像素的宽度的 padding，填充0用于卷积输出原尺寸结果
    卷积后输出尺寸：HxW
    """

    def __init__(self, num_filters: int):
        self.num_filters: int = num_filters

        # Initialize a random filter matrix with num_filters x 3 x 3 dimensions
        # - It contains "num_filters" of 3x3 filters
        #
        # Divided by 9 to reduce the variance of our initial values
        # - Diving by 9 during the initialization is more important than you may think. If the initial values are
        #   too large or too small, training the network will be ineffective.
        # - To learn more, read about: https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks
        self.filters: NDArray = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image: NDArray) -> (NDArray, int, int):
        """
        Iterate the given image (with padding) under filter's size (3x3)
          - image is a 2d numpy array

        Return current region and position info
        """
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region: NDArray = image[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, padded_image: NDArray) -> NDArray:
        """
        Call self.iterate_regions() to go through the image, calculate the given input array with filter, then store the
        result in certain index in output
          - padded_image is a 2d numpy array with size (H+2)x(W+2)
          - output is a 2d numpy array with original size HxW

        Returns a 3d numpy array with dimensions (h, w, num_filters).
        """
        h, w = padded_image.shape
        output: NDArray = np.zeros((h-2, w-2, self.num_filters))

        for region, i, j in self.iterate_regions(padded_image):
            # Calculate current image region with filter matrix (num_filters of 3x3 filters) and get result
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))

        return output

