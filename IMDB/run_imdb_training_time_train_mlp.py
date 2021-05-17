import sys
sys.path.append("..")
from load_data import *
from activation_code_methods import *
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.datasets import imdb
import numpy as np
from keras.utils import to_categorical

# Basic hyper-parameters
batch_size = 512
repeat = 5
begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/models/training_time/model_training_process_'
depth = 1
dataset = 'imdb'

num_classes = 2
weight_decay = 1e-6
lr = 1e-2

width_list = [150] #[60, 80, 100]
output_epoch_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]

training_epoch_list = []
for i in range(len(output_epoch_list)):
    if i == 0:
        training_epoch_list.append(output_epoch_list[i] - 0)
    else:
        training_epoch_list.append(output_epoch_list[i] - output_epoch_list[i-1])

# Load data
##########################################
# number of most-frequent words 
nb_words = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=nb_words)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
def vectorize_sequences(sequences, dimension=nb_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Convert training data to bag-of-words:
x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

# Convert labels from integers to floats:
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
###########################################


input_shape = x_train.shape[1:]

print('dataset: ' + dataset)
print('depth: ' + str(depth))

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        result_list = []

        # building model
        mlp = create_mlp(num_neuron, depth, input_shape, num_classes, weight_decay=weight_decay, bn=False)

        # Compile networks
        #opt = keras.optimizers.Adam(lr=lr)
        mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        mlp.summary()

        mlp.save(save_path + str(0) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1) + '.h5')

        # training according to training epoch list
        for index, training_epoch in enumerate(training_epoch_list):
            # Training policies
            if index == 0:
                def lr_scheduler(epoch):
                    return lr * (0.1 ** (epoch // 20))
            else:
                def lr_scheduler(epoch):
                    return lr * (0.1 ** ((output_epoch_list[index-1] + epoch) // 20))
            reduce_lr = LearningRateScheduler(lr_scheduler)

            # training networks
            mlp.fit(x_train, y_train, batch_size=batch_size, epochs=training_epoch, verbose=1)

            mlp.save(save_path + str(output_epoch_list[index]) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1) + '.h5')