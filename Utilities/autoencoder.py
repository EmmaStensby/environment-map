import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Autoencoder:
    def __init__(self, image_dim):
        self.encoding_dim = 32
        self.image_dim = image_dim

        input_img = keras.Input(shape=(self.image_dim,))
        #encoded = layers.Dense(self.encoding_dim*4, activation='relu')(input_img)
        #encoded = layers.Dense(self.encoding_dim*2, activation='relu')(encoded)
        encoded = layers.Dense(self.encoding_dim*2, activation='relu')(input_img)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(self.encoding_dim*2, activation='relu')(encoded)
        #decoded = layers.Dense(self.encoding_dim*4, activation='relu')(decoded)
        decoded = layers.Dense(self.image_dim, activation='sigmoid')(decoded)

        self.autoencoder = keras.Model(input_img, decoded)
        
        self.encoder = keras.Model(input_img, encoded)
        encoded_input = keras.Input(shape=(self.encoding_dim,))
        #decoder_layer1 = self.autoencoder.layers[-3]
        decoder_layer2 = self.autoencoder.layers[-2]
        decoder_layer3 = self.autoencoder.layers[-1]
        self.decoder = keras.Model(encoded_input, decoder_layer3(decoder_layer2(encoded_input)))

        self.autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())


    def train(self, data, epochs):
        self.autoencoder.fit(data, data, epochs=epochs, batch_size=1000, shuffle=True)

    def encode(self, data):
        encoded_data = self.encoder.predict(data)
        return encoded_data

    def decode(self, data):
        decoded_data = self.decoder.predict(data)
        return decoded_data

    def process(self, data):
        encoded_data = self.encoder.predict(data)
        decoded_data = self.decoder.predict(encoded_data)
        return decoded_data

    def display(self, original_data, data):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2)
        ax[0].plot(original_data)
        ax[1].plot(data)
        plt.show()

    def display_2(self, original_data, data, data_2):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3)
        ax[0].plot(original_data)
        ax[0].set_ylim(0.0,1.0)
        ax[1].plot(data)
        ax[1].set_ylim(0.0,1.0)
        ax[2].plot(data_2)
        ax[2].set_ylim(0.0,1.0)
        plt.show()

    def save_self(self, path):
        path_1 = path + "_ae"
        path_2 = path + "_e"
        path_3 = path + "_d"
        self.autoencoder.save_weights(path_1)

    def load_self(self, path):
        path_1 = path + "_ae"
        path_2 = path + "_e"
        path_3 = path + "_d"
        self.autoencoder.load_weights(path_1)
        

if __name__ == "__main__":
    imgs = np.array([[x/100 for x in range(100)] for y in range(1000)])
    ae = Autoencoder(100)
    ae.load_self("sample")
    decoded_data = ae.process(imgs)
    ae.display(imgs[0], decoded_data[0])
    ae.train(imgs, 50)
    decoded_data = ae.process(imgs)
    ae.display(imgs[0], decoded_data[0])
    ae.save_self("sample")

    resolution = 5

    l1 = []
    l2 = []
    
    for i in range((resolution*resolution)-1):
        l1.append(((1/(resolution*resolution))*(i+1))/3)
        l2.append((((i+1)*(i+1))/(resolution*resolution*resolution*resolution)))

    print("lin")
    print(l1)
    print("x**2")
    print(l2)

