import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Sequential
import pandas as pd
import datetime


class PerceptronMulticapa:
    def __init__(self, model_tag):
        self.model_tag = model_tag
        self.input = Input(4584)  # ver cuantos inputs tiene
        self.hidden = Dense(3, activation="relu")  # 3 capas ocultas
        self.output = Dense(3, activation="softmax")
        self.model = Sequential()
        self.model.add(self.input)
        self.model.add(self.hidden)
        self.model.add(self.output)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )

    def train(self, x, y, xv, yv):
        tag = self.model_tag + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/fit/" + tag
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        self.model.fit(
            x,
            y,
            validation_data=(xv, yv),
            epochs=32,
            batch_size=1,
            verbose=2,
            callbacks=[tensorboard_callback],
        )
        self.model.save("modelos/model_{0}".format(tag))
        yv_pred = self.model.predict(xv)
        pd.DataFrame(yv_pred).to_csv(
            self.model_tag + "_yv_pred.csv", index=False, sep=";"
        )
        pd.DataFrame(yv).to_csv(self.model_tag + "_yv.csv", index=False, sep=";")
