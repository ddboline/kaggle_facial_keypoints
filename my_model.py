#!/usr/bin/python
from __future__ import print_function
import os
os.sys.setrecursionlimit(10000)

import numpy as np

from load_fn import load2d
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator

from collections import OrderedDict
from sklearn.base import clone

import lasagne.layers.cuda_convnet as cuda_convnet

from sklearn.metrics import mean_squared_error

from theano import theano

# use the cuda-convnet implementations of conv and max-pool layer
Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
    
    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print('Early Stopping.')
            print('Best valid loss was {:.6f} at epoch {}.'.format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()


def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch-1])
        getattr(nn, self.name).set_value(new_value)


class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices,:,:,::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices,::2] = yb[indices,::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

def get_neural_network():
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', Conv2DLayer),
            ('pool3', MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),],
        input_shape=(None, 1, 96, 96),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
        dropout3_p=0.3,
        hidden4_num_units=1000,
        dropout4_p=0.5,
        hidden5_num_units=1000,
        output_num_units=30, output_nonlinearity=None,

        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float32(0.9)),

        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
        on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                        AdjustVariable('update_momentum', start=0.9, stop=0.999),
                        EarlyStopping(patience=200),],
        max_epochs=10000,
        verbose=1,)
    return net

SPECIALIST_SETTINGS = [dict(columns=('left_eye_center_x', 'left_eye_center_y',
                                     'right_eye_center_x', 'right_eye_center_y'),
                            flip_indices=((0, 2),(1, 3)),),
                       dict(columns=('nose_tip_x', 'nose_tip_y'), flip_indices=(),),
                       dict(columns=('mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'),
                            flip_indices=(),),
                       dict(columns=('left_eye_inner_corner_x', 'left_eye_inner_corner_y',
                                    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                                    'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
                                    'right_eye_outer_corner_x', 'right_eye_outer_corner_y',),
                            flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),),
                       dict(columns=('left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
                                     'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
                                     'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
                                     'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',),
                            flip_indices=((0, 2), (1, 3), (4, 6), (5, 7),),),]

def fit_specialists(specialist_index=0):
    setting = SPECIALIST_SETTINGS[specialist_index]
    
    cols = setting['columns']
    X, y = load2d(cols=cols)

    model = get_neural_network()
    model.output_num_units = y.shape[1]
    model.batch_iterator_train.flip_indices = setting['flip_indices']
    # set number of epochs relative to number of training examples:
    model.max_epochs = int(1e7 / y.shape[0])
    if 'kwargs' in setting:
        # an option 'kwargs' in the settings list may be used to
        # set any other parameter of the net:
        vars(model).update(setting['kwargs'])

    print("Training model for columns {} for {} epochs".format(
        cols, model.max_epochs))
    model.fit(X, y)
    
    with open('net_specialist_%d.pickle' % specialist_index, 'wb') as f:
        pickle.dump((model, cols) , f, -1)

    return model


def predict_specialists():
    X = load2d(test=True)[0]
    y_pred = np.empty((X.shape[0], 0))
    
    columns = ()
    for idx in range(len(SPECIALIST_SETTINGS)):
       with open('net_specialist_%d.pickle' % specialist_index, 'rb') as f:
           model, cols = pickle.load(specialist_index)
           y_pred1 = model.predict(X)
           y_pred = np.hstack([y_pred,y_pred1])
           columns += cols
    
    y_pred2 = y_pred * 48 + 48
    y_pred2 = y_pred2.clip(0, 96)
    df = DataFrame(y_pred2, columns=columns)
    lookup_table = read_csv('IdLookupTable.csv')
    values = []
    
    for index, row in lookup_table.iterrows():
        values.append((row['RowId'],
                       df.ix[row.ImageId-1][row.FeatureName],))
    
    submission = DataFrame(values, columns=('RowId', 'Location'))
    dstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    filename = 'submission_%s.csv' % dstr
    submission.to_csv(filename, index=False)


def run_full():
    net = get_neural_network()
    
    X, y = load2d()  # load 2-d data
    net.fit(X, y)

    # Training for 1000 epochs will take a while.  We'll pickle the
    # trained model so that we can load it back later:
    import cPickle as pickle
    with open('net.pickle', 'wb') as f:
        pickle.dump(net, f, -1)

    print('mean_squared_error', mean_squared_error(net.predict(X), y))
    return net

def plot_training(net, label='net'):
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pl.plot(train_loss, linewidth=3, label="train")
    pl.plot(valid_loss, linewidth=3, label="valid")
    pl.grid()
    pl.legend()
    pl.xlabel("epoch")
    pl.ylabel("loss")
    #pl.ylim(1e-3, 1e-2)
    pl.yscale("log")
    pl.savefig('training_loss_net.png')

if __name__ == '__main__':
    index = -1
    for arg in os.sys.argv:
        try:
            index = int(arg)
            break
        except ValueError:
            continue
    if index == -1:
        net = run_full()
        plot_training(net)
    else:
        fit_specialists(specialist_index=index)
