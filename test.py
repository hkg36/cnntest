import numpy as np

a = np.array([[10, 30, 20], [60, 40, 50]])
ai = np.argsort(a, axis=1)
print(ai)
ai=np.array([2,2])
ai=np.expand_dims(ai,axis=1)
res=np.take_along_axis(a, ai, axis=1)


# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
image = image.imread('data/img.jpg')
image=np.moveaxis(image, 2, 0)
# summarize shape of the pixel array
# display the array of pixels as an image
pyplot.imshow(image[1],cmap ='gray')
pyplot.show()