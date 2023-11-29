import productionplot
from rnojd_corica import CoroICARNOJD
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import FastICA
from rnojd import *
import cv2
import numpy


def scale_image(image):
    return ((image - np.min(image)) / (np.max(image) - np.min(image)))


def vectorize_image(image, grid):
    gridvals = np.unique(grid)
    vecimg = np.empty(0)
    for val in gridvals:
        vecimg = np.concatenate([vecimg, image[grid == val].ravel()])
    return vecimg


def grid_to_indices(grid):
    gridvals = np.unique(grid)
    image = np.empty(np.shape(grid) + (3, ), dtype=int)
    start = 0
    for layer in [0, 1, 2]:
        for val in gridvals:
            end = start + np.sum(grid == val)
            image[grid == val, layer] = np.arange(start, end)
            start = end
    return image


def construct_grid(vpixel, hpixel, vnum, hnum):
    return np.arange(vnum * hnum).reshape(
        vnum, hnum).repeat(
            vpixel // vnum, axis=0).repeat(
                hpixel // hnum, axis=1)


def load_image_vectorized(fname, grid):
    image = Image.open(fname)
    img = np.asarray(image, dtype='float')
    img = cv2.resize(img, (450,300),)
    #print(img.shape)
    return np.vstack([vectorize_image(img[..., k], grid)
                      for k in range(3)])


def scale_to_closest_ground_truth(img, images):
    img = img.reshape(*images[0].shape)
    curbest = [None, np.inf]
    for true_img in images:
        newimg = np.copy(img)
        for row in range(3):
            newimg[row, :] *= true_img[row, :].dot(newimg[row, :]) / \
                newimg[row, :].dot(newimg[row, :])
        ss = ((newimg - true_img)**2).sum()
        if ss < curbest[1]:
            curbest = [newimg, ss]
    return curbest[0].reshape(-1)


def image_dist(imgA, imgB):
    score = 3
    for k in range(3):
        score -= np.abs(np.corrcoef(
            imgA[..., k].ravel(),
            imgB[..., k].ravel())[0, 1])
    return score

def rnojd_wrapper(X, trials=3):
    return rnojd(X, trials = trials,pd=False).T

def rffdiag_wrapper(X, pd=True, max_iter = 100):
    return ffdiag_rnojd(X,pd=pd,max_iter=max_iter)[0].T


# Load images
grid = construct_grid(300, 450, 30, 45)
grid_re = grid_to_indices(grid)
image_indices = [3,4,5,6]
images = [load_image_vectorized('images/pic{}_middle.jpg'.format(fname),
                                grid)
          for fname in image_indices]
num_images = len(images)

# Construct source matrix
# rows are the different images
# pixels are unravelled in the other dimension
# one environment = color channel after the other
# i.e. row 1 has 405000 pixels, the first third is belongs to
# the first color channel and so on
Smat = np.vstack([img.reshape(-1)
                  for img in images])

envlen = images[0].shape[1]

np.random.seed(123)
A = np.random.normal(0, 1, (4, 4))

# Construct noise (8 dimensional noise, changes per color channel)
Hfull = np.hstack([np.repeat(noise, 4, axis=0)
                   for noise in [
                       np.vstack([
                           np.random.uniform(-10, 0, (1, envlen)),
                           np.random.uniform(-30, 0, (1, envlen))]),
                       np.vstack([
                           np.random.uniform(0, 20, (1, envlen)),
                           np.random.uniform(0, 20, (1, envlen))]),
                       np.vstack([
                           np.random.uniform(-40, 40, (1, envlen)),
                           np.random.uniform(-20, 20, (1, envlen))])]])
# Add spatial aka "time" dependence
deplag = 50
depmat = 1.5 / (np.arange(1, deplag + 1)**2)
for group in range(len(images) - 1):
    for t in range(group * envlen,
                   (group + 1) * envlen - 2 * deplag):
        Hfull[:, (t + deplag)] = depmat.dot(
            Hfull[:, (t + deplag):(t + 2 * deplag)].T)
Hmat = np.random.normal(0, 1, (4, 8)).dot(Hfull)

Xmat = A.dot(Smat + Hmat)
images_g0 = [scale_image((Smat + Hmat)[imgind, :][grid_re])
             for imgind in range(4)]
images_g0_vectorized = [np.vstack([vectorize_image(img[..., k], grid)
                                   for k in range(3)])
                        for img in images_g0]

print(Xmat.shape)
# Apply coroICA, fastICA, uwedgeICA
group_index = np.repeat(np.arange(0, 3), envlen)

timelags = [0,1,2]
partitionsize = 900
rnojd_ica =  CoroICARNOJD(partitionsize=partitionsize,
                      timelags=timelags,
                      instantcov=False,
                      pairing='neighbouring',
                      max_matrices=0.2)
rffiag_ica = CoroICARNOJD(partitionsize=partitionsize,
                      timelags=timelags,
                      instantcov=False,
                      pairing='neighbouring',
                      max_matrices=0.2)
uwedgeica = CoroICARNOJD(partitionsize=partitionsize,
                      timelags=timelags,
                      instantcov=False,
                      pairing='neighbouring',
                      max_matrices=0.2,)
rnojd_ica.fit(Xmat.T, use_uwedge=True, random_uwedge=False)
rffiag_ica.fit(Xmat.T, algorithm=rffdiag_wrapper)

uwedgeica.fit(Xmat.T, use_uwedge=True, random_uwedge=True)
unmixings = [
     rnojd_ica.V_,  rffiag_ica.V_,  uwedgeica.V_,]
# Plot example noisy image
plt.figure()
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(scale_image(Smat[0, :][grid_re]))
plt.title('Raw image')
plt.subplot(1, 3, 2)
plt.axis('off')
plt.title('Noisy variant')
plt.imshow(scale_image((Smat )[0, :][grid_re]))
plt.subplot(1, 3, 3)
plt.axis('off')
plt.title('After mixing')
plt.imshow(scale_image(Xmat[0, :][grid_re]))

# Plot
images_per_method = [images_g0] +  \
    [[scale_image(
        scale_to_closest_ground_truth(V.dot(Xmat)[imgind, :],
                                      images_g0_vectorized)[grid_re])
      for imgind in range(len(images))]
     for V in [np.eye(*unmixings[0].shape)] + unmixings]
cols = ['Image {}'.format(col) for col in range(1, num_images+1)]
rows = ['ground truth', 'observation', 'rsdc', 'rffdiag', 'uwedge',]
fig = plt.figure(dpi=600, figsize=(productionplot.TEXTWIDTH, 5))

for mind, method in enumerate(rows):
    subplotind = 1
    for imgind in range(num_images):
        ax = fig.add_subplot(6, num_images, subplotind+mind*num_images)
        targetimg = images_per_method[mind][imgind]
        ax.imshow(scale_image(targetimg))
        ax.tick_params(axis='x', which='both',
                       bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both',
                       left=False, right=False, labelleft=False)
        if mind == 0:
            ax.set_title(cols[imgind])
        if imgind == 0:
            ax.set_ylabel(rows[mind], rotation=90)
        subplotind += 1
fig.savefig('picture_example_original.pdf', bbox_inches='tight')

#Running time report
repeats = 1
times_rnojd, times_rnojd_manopt, times_uwedge, times_ffdiag = 0, 0, 0, 0
for _ in range (repeats):
    _, time_rnojd = uwedgeica.fit(Xmat.T, use_uwedge=True, random_uwedge=True)
    _, time_rnojd_manopt = rffiag_ica.fit(Xmat.T, algorithm=rffdiag_wrapper)
    _, time_uwedge = uwedgeica.fit(Xmat.T, use_uwedge=True, random_uwedge=False)
    times_rnojd += time_rnojd
    times_rnojd_manopt += time_rnojd_manopt
    times_uwedge += time_uwedge
print("RNOJD time: ", 1000*times_rnojd/repeats)
print("RNOJD_Manopt time: ", 1000*times_rnojd_manopt/repeats)
print("uwedge time: ", 1000*times_uwedge/repeats)