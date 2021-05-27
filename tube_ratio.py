from load_data import *
from activation_code_methods import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import load_model

mnist_path = '/public/data1/users/leishiye/datasets/mnist.npz'
(x_train, y_train), (x_test, y_test) = load_mnist(path=mnist_path, flatten=True)

load_path = '/public/data1/users/leishiye/neural_code/models/model_mnist/layer_width/model_layer_width_100_mnist_depth_1_iter4.h5'
mlp = load_model(load_path)

train_activation_codes, test_activation_codes = compute_activation_code_for_mlp(x_train, x_test, model=mlp)

train_label_scalar = np.argmax(y_train, axis=1).squeeze()
test_label_scalar = np.argmax(y_test, axis=1).squeeze()

sub_code = np.sum(sub_code, axis=1)

for i in range(10):
    sub_code = train_activation_codes[train_label_scalar==i]
    print(sub_code.shape)
    sub_code = np.sum(sub_code, axis=0)
    print(sub_code)
    print(np.min(a))