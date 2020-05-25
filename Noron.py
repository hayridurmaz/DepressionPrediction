import numpy as np


class Noron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # Nöron içerisinde yapılan toplam işlemi sonrası
    # değerimizi 0 ile 1 arasında standartlaştırmak için
    # sigmoid metodumuzu oluşturuyoruz
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    ## Sigmoid fonksiyonunun türevi
    def sigmoid_derivative(self, x):
        # Sigmoid fonksiyonunun türevi: f'(x) = f(x) * (1 - f(x))
        sig = self.sigmoid(x)
        result = sig * (1 - sig)
        return result

    def mse_loss(self, y_real, y_prediction):
        # y_real ve y_prediction aynı boyutta numpy arrayleri olmalıdır.
        return ((y_real - y_prediction) ** 2).mean()

    def feedforward(self, data):
        # Girdilerin ağırlıkla ve ardından bias ile toplama işlemi
        sumResult = (np.dot(self.weights, data) + self.bias)

        # En son olarak sonucu sigmoid fonksiyonundan geçirerek
        # 0 ile 1 arasında standartlaştırılmış sonuç değerini elde ediyoruz
        return self.sigmoid(sumResult)



