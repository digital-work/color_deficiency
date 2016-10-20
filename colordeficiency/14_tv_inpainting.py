from pylab import *

ion()

im = imread('lena.png')
im = sum(im, 2) / 3.
im = zeros((30,30))
for i in range(30):
    for j in range(i):
        im[i,j]=1
mask = zeros(shape(im))
#mask[295:315,300:320] = 1
mask[10:20,10:20] = 1
ind = (mask == 1)
im[im > 1] = 1
im[im < 0] = 0
im[ind] = .5
im0 = im.copy()
data = imshow(im, cm.gray)
draw()

dt = .02
eps = .001

while True:
    gx, gy = gradient(im)
    gradnorm = sqrt(gx**2 + gy**2) + eps
    gxx,tmp = gradient(gx / gradnorm)
    tmp, gyy = gradient(gy / gradnorm)
    tv = gxx + gyy
    im[ind] = im[ind] + dt * tv[ind]
    data.set_array(im)
    draw()
