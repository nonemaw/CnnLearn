import mnist

from conv import Conv3x3

# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
train_images = mnist.train_images()
train_labels = mnist.train_labels()

conv = Conv3x3(num_filters=8)
output = conv.forward(train_images[0])
print(output.shape)  # (26, 26, 8)

"""
在这个3x3卷积中，我们已经预先假设了input是一个二维NDArray，这是因为一个MNIST图像就是以
这种方式存储的

当它作为卷积的第一层时这是能够工作的，但是在真实环境中CNN会有很多不同的卷积层。如果我们需
要建立一个使用多个3x3卷积过滤的CNN网络，那么我们必须提供一个三维的NDArray作为input
"""
