import numpy as np  # helps with the math


class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50]])
        self.error_history = []
        self.epoch_list = []

    # activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def train(self, epochs=25000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction


class Network:

    ## Ağırlıklarımızı ve bias değerlerimizi burada oluşturulmaktadır.
    def __init__(self):

        # Ağ üzerinden 3 adet nöron olduğu için
        # 6 adet ağırlık ve 3 adet bias değeri olmalı
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    ## Sigmoid fonksiyonu
    def sigmoid(self, x):

        # Sigmoid aktivasyon fonksiyonu : f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

    ## Sigmoid fonksiyonunun türevi
    def sigmoid_turev(self, x):

        # Sigmoid fonksiyonunun türevi: f'(x) = f(x) * (1 - f(x))
        sig = self.sigmoid(x)
        result = sig * (1 - sig)

        return result

    def mse_loss(self, y_real, y_prediction):

        # y_real ve y_prediction aynı boyutta numpy arrayleri olmalıdır.
        return ((y_real - y_prediction) ** 2).mean()

    ## İleri beslemeli nöronlar üzerinden tahmin
    ## değerinin elde edilmesi

    def feedforward(self, row):

        # h1 nöronunun değeri
        h1 = self.sigmoid((self.w1 * row[0]) + (self.w2 * row[1]) + self.b1)

        # h2 nöronunun değeri
        h2 = self.sigmoid((self.w3 * row[0]) + (self.w4 * row[1]) + self.b2)

        # Tahmin değeri 01 nöronun değeri
        o1 = self.sigmoid((self.w5 * h1) + (self.w6 * h2) + self.b3)

        return o1

        ## Belitiler iteresyon sayısı kadar model eğitimi

    def train(self, data, labels):

        learning_rate = 0.001
        epochs = 10000

        for epoch in range(epochs):

            for x, y in zip(data, labels):
                # Neuron H1
                sumH1 = (self.w1 * x[0]) + (self.w2 * x[1]) + self.b1
                H1 = self.sigmoid(sumH1)

                # Neuron H2
                sumH2 = (self.w3 * x[0]) + (self.w4 * x[1]) + self.b2
                H2 = self.sigmoid(sumH2)

                # Neuron O1
                sumO1 = (self.w5 * H1) + (self.w6 * H2) + self.b3
                O1 = self.sigmoid(sumO1)

                # Tahmin değerimiz
                prediction = O1

                # Türevlerin Hesaplanması
                # dL/dYpred :  y = doğru değer | prediciton: tahmin değeri
                dLoss_dPrediction = -2 * (y - prediction)

                # Nöron H1 için ağırlık ve bias türevleri
                dH1_dW1 = x[0] * self.sigmoid_turev(sumH1)
                dH1_dW2 = x[1] * self.sigmoid_turev(sumH1)
                dH1_dB1 = self.sigmoid_turev(sumH1)

                # Nöron H2 için ağırlık ve bias türevleri
                dH2_dW3 = x[0] * self.sigmoid_turev(sumH2)
                dH2_dW4 = x[1] * self.sigmoid_turev(sumH2)
                dH2_dB2 = self.sigmoid_turev(sumH2)

                # Nöron O1 (output) için ağırlık ve bias türevleri
                dPrediction_dW5 = H1 * self.sigmoid_turev(sumO1)
                dPrediction_dW6 = H1 * self.sigmoid_turev(sumO1)
                dPrediction_dB3 = self.sigmoid_turev(sumO1)

                # Aynı zamanda tahmin değerinin H1 ve H2'ye göre türevlerinin de
                # hesaplanması gerekmektedir.
                dPrediction_dH1 = self.w5 * self.sigmoid_turev(sumO1)
                dPrediction_dH2 = self.w6 * self.sigmoid_turev(sumO1)

                ## Ağırlık ve biasların güncellenmesi

                # H1 nöronu için güncelleme
                self.w1 = self.w1 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW1)
                self.w2 = self.w2 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW2)
                self.b1 = self.b1 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dB1)

                # H2 nöronu için güncelleme
                self.w3 = self.w3 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW3)
                self.w4 = self.w4 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW4)
                self.b2 = self.b2 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dB2)

                # O1 nöronu için güncelleme
                self.w5 = self.w5 - (learning_rate * dLoss_dPrediction * dPrediction_dW5)
                self.w6 = self.w6 - (learning_rate * dLoss_dPrediction * dPrediction_dW6)
                self.b3 = self.b3 - (learning_rate * dLoss_dPrediction * dPrediction_dB3)

            predictions = np.apply_along_axis(self.feedforward, 1, data)
            loss = self.mse_loss(labels, predictions)
            print("Devir %d loss: %.7f" % (epoch, loss))

