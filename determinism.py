from GNEMS.GNEMS import GraphicallyGuidedEMSegmentor, GNEMS_segment
import matplotlib.pyplot as plt
import numpy as np

# load image
img = plt.imread('../datasets/clouds/images/0007.png')
img = img[:512,:512,0]
seg1 = GNEMS_segment(img, d=16, n_filters=16, dropout=0, lambda_=0.3, lr=0.001, iterations=25, subset_size=0.5, prediction_stride=32, slic_segments=100, sigma=3, deterministic=True, show_progress=False)
seg2 = GNEMS_segment(img, d=16, n_filters=16, dropout=0, lambda_=0.3, lr=0.001, iterations=25, subset_size=0.5, prediction_stride=32, slic_segments=100, sigma=3, deterministic=True, show_progress=False)
seg3 = GNEMS_segment(img, d=16, n_filters=16, dropout=0, lambda_=0.3, lr=0.001, iterations=25, subset_size=0.5, prediction_stride=32, slic_segments=100, sigma=3, deterministic=True, show_progress=False)

# show image and all segmentations
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(2,2,2)
plt.imshow(seg1, cmap='gray')
plt.subplot(2,2,3)
plt.imshow(seg2, cmap='gray')
plt.subplot(2,2,4)
plt.imshow(seg3, cmap='gray')
plt.show()