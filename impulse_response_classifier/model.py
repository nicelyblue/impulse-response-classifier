import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, f1_score

class IRClassifier:
    def __init__(self, input_shape, model_path=os.path.relpath('model'), n_classes=2):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model_path = model_path
        self.build_model()

    def build_model(self):
        input = tf.keras.layers.Input(
            shape=self.input_shape[0])

        dense = tf.keras.layers.Dense(256, activation='relu')(input)
        # dense = tf.keras.layers.Dropout(0.5)(dense)
        dense = tf.keras.layers.Dense(64, activation='relu')(dense)
        # dense = tf.keras.layers.Dropout(0.5)(dense)
        x = tf.keras.layers.Dense(32, activation='relu')(dense)
        # x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)

        self.model = tf.keras.Model(
            inputs=input, outputs=output)

    def train(self, train_data, val_data, epochs, early_stop_patience=5, reduce_lr_factor=0.5, reduce_lr_patience=2, reduce_lr_min_delta=0.01):
        self.model.summary(expand_nested=True)
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                           loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy', patience=early_stop_patience, restore_best_weights=True)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=reduce_lr_factor,
                                                         patience=reduce_lr_patience, min_delta=reduce_lr_min_delta)

        history = self.model.fit(train_data, epochs=epochs,
                                 validation_data=val_data, callbacks=[early_stop, reduce_lr])

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model.save(self.model_path)

        return history

    def evaluate(self, test_data):
        model = tf.keras.models.load_model(self.model_path)
        y_true = test_data.labels()
        y_pred = np.argmax(model.predict(test_data), axis=-1)
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        selectivity = 1 - recall
        f1 = f1_score(y_true, y_pred, average='macro')

        with open(os.path.join(self.model_path, 'eval.txt'), 'w') as f:
            f.write("Confusion matrix:\n{}\n".format(cm))
            f.write("Accuracy: {}\n".format(acc))
            f.write("Balanced accuracy: {}\n".format(balanced_acc))
            f.write("Recall: {}\n".format(recall))
            f.write("Selectivity: {}\n".format(selectivity))
            f.write("F1 score: {}\n".format(f1))

        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'confusion_matrix.png'))
        plt.close()
