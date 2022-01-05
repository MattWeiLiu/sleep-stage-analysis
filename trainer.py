from source.configuration import Config
from model.base_rnn import baseRNN

import os, sys
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

def main():
    # Import Configuration
    config = Config(os.path.join(os.getcwd(), "config/{}".format(sys.argv[1])))
    
    # Import Data
    inputs, targets = [], []

    for npz_path in glob(config.data.sleep_features_path + '*'):
        data = np.load(npz_path)
        inputs.append(data['data'])
        targets.append(data['label'])

    inputs = np.array(inputs)
    targets = np.array(targets)
    
    ## early stopping
    if config.train.early_stop:
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=config.train.patience)
    else:
        callback = None
        
    # loss
    if config.model.loss == 'CategoricalCrossentropy':
        loss = tf.keras.losses.CategoricalCrossentropy()

    if config.train.cross_validation:
        kfold = KFold(n_splits=config.train.num_folds, random_state=None)

        fold_no = 1
        for train_i, test_i in kfold.split(inputs):

            model = baseRNN(activation=config.model.activation, 
                            L1=float(config.model.l1_scale), 
                            L2=float(config.model.l2_scale), 
                            feature_dims=config.data.feature_dims, 
                            lstm_nodes=config.model.lstm_nodes, 
                            dense_nodes=config.model.dense_nodes, 
                            window_size=config.data.window_size, 
                            keep_drop=config.model.keep_drop, 
                            output_dim=config.model.output_dim)

            # optimizer & learning rate
            if config.train.optimizer.method == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=float(config.train.optimizer.learning_rate))
            elif config.train.optimizer.method == 'RMSprop':
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=float(config.train.optimizer.learning_rate))

            model.compile(loss=loss, 
                          optimizer=optimizer, 
                          metrics=["accuracy"])

            X_train, y_train, X_test, y_test = inputs[train_i], targets[train_i], inputs[test_i], targets[test_i]
            X_train, y_train, X_test, y_test = np.concatenate(X_train), np.concatenate(y_train), np.concatenate(X_test), np.concatenate(y_test)
            
            # normalize
            norm_min = X_train.min(axis=1).min(axis=0)
            norm_max = X_train.max(axis=1).max(axis=0)
            X_train = (X_train - norm_min)/(norm_max - norm_min)
            X_test  = (X_test - norm_min)/(norm_max - norm_min)
            
            # fill nan as 0
            X_train[np.isnan(X_train)] = 0
            X_test[np.isnan(X_test)]   = 0
            
            # label one hot encoding
            y_train = np.eye(6)[y_train-11]
            y_test  = np.eye(6)[y_test-11]

            print('------------------------------------------------------------------------')
            print('Training for fold {} ...'.format(fold_no))
            history = model.fit(X_train, 
                                y_train, 
                                batch_size=config.train.batch_size, 
                                epochs=config.train.epochs, 
                                callbacks=[callback], 
                                verbose=2, 
                                validation_data=(X_test, y_test))
            
            # save model
            if config.save.model:
                if not os.path.exists('save/model/{}/'.format(sys.argv[1][:-5])):
                    os.mkdir('save/model/{}/'.format(sys.argv[1][:-5]))
                model.save_weights('save/model/{}/model_{}.h5'.format(sys.argv[1][:-5], fold_no))
            
            # visualization 
            if config.save.img:
                if not os.path.exists('save/visualization/{}/'.format(sys.argv[1][:-5])):
                    os.mkdir('save/visualization/{}/'.format(sys.argv[1][:-5]))
                acc = [n*100 for n in history.history['accuracy']]
                val_acc = [n*100 for n in history.history['val_accuracy']]
                loss_history = history.history['loss']
                val_loss_history = history.history['val_loss']

                plt.figure(figsize=(20,5))
                plt.plot(acc, color=[255/255, 0, 0], label='accuracy')
                plt.plot(val_acc, color=[100/255, 0, 0], label='val_accuracy')
                plt.legend(loc='center right')
                plt.savefig('save/visualization/{}/img_acc_{}.jpg'.format(sys.argv[1][:-5], fold_no), dpi=100)
                plt.show()

                plt.figure(figsize=(20,5))
                plt.plot(loss_history, color=[0, 0, 255/255], label='loss')
                plt.plot(val_loss_history, color=[0,0, 100/255], label='val_loss')
                plt.legend(loc='center right')
                plt.savefig('save/visualization/{}/img_loss_{}.jpg'.format(sys.argv[1][:-5], fold_no), dpi=100)
                
            # Increase fold number
            fold_no += 1
        
    else:
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)
        X_train, y_train, X_test, y_test = np.concatenate(X_train), np.concatenate(y_train), np.concatenate(X_test), np.concatenate(y_test)

        # normalize
        norm_min = X_train.min(axis=1).min(axis=0)
        norm_max = X_train.max(axis=1).max(axis=0)
        X_train = (X_train - norm_min)/(norm_max - norm_min)
        X_test  = (X_test - norm_min)/(norm_max - norm_min)
        
        # fill nan as 0
        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)]   = 0
        
        # label one hot encoding
        y_train = np.eye(6)[y_train-11]
        y_test  = np.eye(6)[y_test-11]

        model = baseRNN(activation=config.model.activation, 
                        L1=float(config.model.l1_scale), 
                        L2=float(config.model.l2_scale), 
                        feature_dims=config.data.feature_dims, 
                        lstm_nodes=config.model.lstm_nodes, 
                        dense_nodes=config.model.dense_nodes, 
                        window_size=config.data.window_size, 
                        keep_drop=config.model.keep_drop, 
                        output_dim=config.model.output_dim)

        # optimizer & learning rate
        if config.train.optimizer.method == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=float(config.train.optimizer.learning_rate))
        elif config.train.optimizer.method == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=float(config.train.optimizer.learning_rate))

        model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=["accuracy"])

        print('------------------------------------------------------------------------')
        print('Training ...')
        history = model.fit(X_train, 
                            y_train, 
                            batch_size=config.train.batch_size, 
                            epochs=config.train.epochs, 
                            callbacks=[callback], 
                            verbose=2, 
                            validation_data=(X_test, y_test))
        
        # save model
        if config.save.model:
            if not os.path.exists('save/model/{}/'.format(sys.argv[1][:-5])):
                os.mkdir('save/model/{}/'.format(sys.argv[1][:-5]))
            model.save_weights('save/model/{}/model.h5'.format(sys.argv[1][:-5]))
        
        # visualization 
        if config.save.img:
            if not os.path.exists('save/visualization/{}/'.format(sys.argv[1][:-5])):
                os.mkdir('save/visualization/{}/'.format(sys.argv[1][:-5]))
            acc = [n*100 for n in history.history['accuracy']]
            val_acc = [n*100 for n in history.history['val_accuracy']]
            loss_history = history.history['loss']
            val_loss_history = history.history['val_loss']

            plt.figure(figsize=(20,5))
            plt.plot(acc, color=[255/255, 0, 0], label='accuracy')
            plt.plot(val_acc, color=[100/255, 0, 0], label='val_accuracy')
            plt.legend(loc='center right')
            plt.savefig('save/visualization/{}/img_acc.jpg'.format(sys.argv[1][:-5]), dpi=100)
            plt.show()

            plt.figure(figsize=(20,5))
            plt.plot(loss_history, color=[0, 0, 255/255], label='loss')
            plt.plot(val_loss_history, color=[0,0, 100/255], label='val_loss')
            plt.legend(loc='center right')
            plt.savefig('save/visualization/{}/img_loss.jpg'.format(sys.argv[1][:-5]), dpi=100)
        
if __name__ == '__main__':
    main()
