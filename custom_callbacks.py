'''
Some custom callback function to strengthen up training code and tensorboard
'''
import numpy as np
import keras
import tensorflow as tf
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
class loss_history(keras.callbacks.Callback):
    """
    Record loss history by step in Tensorboard
    """
    def __init__(self, model, tensorboard, names=['acc', 'loss']):
        self.model = model
        self.tensorboard = tensorboard
        self.names = names

    def on_train_begin(self, logs={}):
        self.step = 0

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for name in self.names:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = logs[name]
            summary_value.tag = name+'_step'
            self.tensorboard.writer.add_summary(summary, self.step)
            self.tensorboard.writer.flush()
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


class evaluate_step(keras.callbacks.Callback):
    """
    Custom callback function to enable evaluation per step
    """
    def __init__(self, model, checkpointer, tensorboard, evaluate_every, batch_size, 
                 x_dev, y_dev):
        self.model = model
        self.evaluate_every = evaluate_every
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.batch_size = batch_size
        self.checkpointer = checkpointer
        self.tensorboard = tensorboard
        self.max_step = 0

    def on_train_begin(self, logs={}):
        self.step = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        if self.step % self.evaluate_every == 0:
            a= self.x_dev
            b=self.y_dev
            logs = self.model.evaluate(x=self.x_dev, y=self.y_dev, batch_size=self.batch_size, verbose=0)
            y_pred2 = self.model.predict(self.x_dev).ravel()
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(self.y_dev, y_pred2)
            auc_keras = auc(fpr_keras, tpr_keras)

            if self.checkpointer.monitor_op(logs[1], self.checkpointer.best):
                self.checkpointer.best = logs[1]
                self.max_step = self.step
                path = 'logs/checkpoints/vdcnn_weights_val_acc_%0.4f.h5' % (self.checkpointer.best)
                if self.checkpointer.save_weights_only:
                    self.model.save_weights(path, overwrite=True)
                else:
                    self.model.save(path, overwrite=True)
                time_str = datetime.datetime.now().isoformat()
                print()
                print("auc is",auc_keras)
                print("{}: Saving model with val_acc {:g}, at step {}, epoch {}.".format(time_str, self.checkpointer.best, self.max_step, self.epoch+1))
                print()
            if self.tensorboard is not None:
                names = ['val_loss_step', 'val_acc_step']
                for i in range(len(names)):
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = logs[i]
                    summary_value.tag = names[i]
                    self.tensorboard.writer.add_summary(summary, self.step)
                    self.tensorboard.writer.flush()

class TestCallback(keras.callbacks.Callback):
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        y_pred = self.model.predict(x)
        y_pred2 = self.model.predict(x).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred2)
        auc_keras = auc(fpr_keras, tpr_keras)
        print("auc is ", auc_keras)
        y_pred = (y_pred > 0.5)
        print(y_pred)
        cm = confusion_matrix(y, y_pred)
        prec= precision_score(y, y_pred)
        recall=recall_score(y, y_pred)
        print(cm)
        print(prec)

        #loss, acc = self.model.evaluate(x, y, verbose=0)
        #print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        #aa=np.argmax(y, axis=1)
        # pred = self.model.predict(x, verbose=0)
        # pred2= np.argmax(pred, axis=1)
        # y_compare= np.argmax(y, axis= 0)
        # print(pred)
        # print("/n")
        # print(y_compare)
        # #score= metrics.accuracy_score(y_compare, pred)
        # #print("final accuracy ois ", score)
        #
        # ###########
        #
        # cm= confusion_matrix(y_compare, pred)
        # np.set_printoptions(precision=2)
        # print("con withput normalization")
        # print(cm)
        #plt.figure()
        #plot_confusion_matrix(cm,y)
        # pred_prob= self.model.predict_proba(x, verbose=0)
        #pred = np.array(pred)

        # print(metrics.confusion_matrix(x, pred))
        # print(" this is pres", pred)
        # print("/n")
        # print(" this is pres", y_classes)
        # print("/n")
        # print('True', y.values[0:25])
        # print('Pred', pred[0:25])

        #p_classes = []
        #
        #true1=0
        #false1=0
        # for p in pred:
        #     if p < 0.5:
        #         p_classes.append(0)
        #         true1+=1
        #     else:
        #         p_classes.append(1)
        #         false1+=1
        # print("true is", true1)
        # print("false is ", false1)
        # tn, fp, fn, tp = confusion_matrix(y, pred)
        # accuracy = (tp + tn) / (tp + tn + fp + fn)
        # print("accuracy: {}".format((tp + tn) / (tp + tn + fp + fn)))
        # print("precision : {:.4f} / {:.4f}".format(tp / (tp + fp), tn / (fn + tn)))
        # print("recall : {:.4f} / {:.4f}".format(tp / (tp + fn), tn / (fp + tn)))
        # print("F1 score : {:.4f} / {:.4f}".format(2 * tp / (2 * tp + fp + fn), 2 * tn / (2 * tn + fp + fn)))
        # auc = roc_auc_score(pred_prob, pred)
        # print("auc is", auc)

        print(recall)