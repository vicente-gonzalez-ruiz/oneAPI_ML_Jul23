import numpy as np

################################################################
#
# Import dptcl and patch here

import dpctl
print("dpctl.__version__ = ", dpctl.__version__)

from sklearnex import patch_sklearn
patch_sklearn()
#
################################################################

X = np.array([[1,  2], [1,  4], [1,  0],
              [10, 2], [10, 4], [10, 0]])

# You need to re-import scikit-learn algorithms after the patch
from sklearn.cluster import KMeans

device = dpctl.select_default_device()
print("Using device ...")
device.print_device_info()

x_device = dpctl.tensor.asarray(X,  device = device)
#x_device = dpctl.tensor.to_numpy(X,  device = device)

kmeans = KMeans(n_clusters=2, init='random', random_state=0).fit(x_device)
print(f"kmeans.labels_ = {kmeans.labels_}")
