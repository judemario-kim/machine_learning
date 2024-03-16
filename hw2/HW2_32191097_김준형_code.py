import numpy as np
import matplotlib.pyplot as plt
import csv

f = open('hw2_data.csv', 'r', encoding='utf-8') #파일 디렉토리 변경시 수정 필요
data_f = csv.reader(f)

xs = []
ys = []

for i, line in enumerate(data_f):
    if i == 0:
      continue
    xs.append(float(line[0]))
    ys.append(float(line[1]))

xs = np.array(xs)
ys = np.array(ys)
f.close()

np.random.seed(42)

class FCNN:
    def __init__(self, layers, active_funcs, epochs, learning_rate = 0.00001, verbose = 1):
        self.layers = layers
        self.b_s = []
        self.z_s = [] # n*1 n*10 n*20 ...
        self.a_s = [] # n*1 n*10 n*20 ...
        self.w_s = [] # 1*10 10*20 20*40 ...
        self.active_funcs = active_funcs
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.validation_err = []
        self.training_err = []

    def training(self, xs, ys):
        x_train, x_val = xs[:int((len(xs)/10)*8)], xs[int((len(xs)/10)*8):]
        y_train, y_val = ys[:int((len(ys)/10)*8)], ys[int((len(ys)/10)*8):]


        self.initialize_weight()
        cost_function = self.get_cost_function()
        count = 0
        for epoch in range(self.epochs):
            for i in range(0, len(x_train), int(len(x_train)*0.2)):
                x_batch = x_train[i:i+int(len(x_train)*0.2)]
                y_batch = y_train[i:i+int(len(y_train)*0.2)]
                self.forwarding(x_batch)
                loss = self.a_s[-1] - y_batch
                self.back_propagation(loss)
            self.forwarding(x_train)
            self.training_err.append(np.mean(cost_function(y_train, self.a_s[-1])))
            self.forwarding(x_val)
            self.validation_err.append(np.mean(cost_function(y_val, self.a_s[-1])))
            if epoch % (self.epochs//100) == 0:
                print(self.training_err[epoch], self.validation_err[epoch])
                print(count,"percent done")
                count+=1

    def predict_and_plot(self, xs, ys):
        self.forwarding(xs)
        plt.plot(xs, ys, 'or')
        plt.plot(xs, self.a_s[-1], 'og')
        plt.show()

    def plot_validerr(self):
        tmp = np.array([i for i in range(len(self.validation_err))])
        plt.plot(tmp, np.array(self.validation_err), 'or', label="validation")
        plt.plot(tmp, np.array(self.training_err), 'og', label="training")
        plt.show()

    def forwarding(self, xs):
        self.z_s[0] = xs
        for i in range(len(self.layers)):
            self.a_s[i] = self.get_activ(self.active_funcs[i])(self.z_s[i])
            if i < len(self.layers) - 1:
                tmp = np.matmul(self.a_s[i], self.w_s[i])
                tmp += self.b_s[i]

                self.z_s[i+1] = tmp
        return

    def back_propagation(self, grad):
        for i in range(len(self.layers)-2, -1, -1):
            grad = grad * self.get_grad(self.active_funcs[i+1])(self.a_s[i+1])
            self.w_s[i] = self.w_s[i] - self.learning_rate * np.matmul(self.z_s[i].T, grad)
            self.b_s[i] = self.b_s[i] - self.learning_rate * grad.sum(axis=0)
            grad = np.matmul(grad, self.w_s[i].T)
        return

    def initialize_weight(self):
        self.a_s = [[] for _ in range(len(self.layers))]
        self.z_s = [[] for _ in range(len(self.layers))]
        self.b_s = []
        tmp = []
        for i in range(len(self.layers)-1):
            tmp2 = []
            for _ in range(self.layers[i]):
                tmp2.append(np.random.rand(self.layers[i+1]))
            tmp.append(np.array(tmp2))
            self.b_s.append(np.random.rand(self.layers[i+1]))
        self.w_s = tmp
        return

    def get_cost_function(self):
        def mse(pred, y):
            return (1/2)*((pred - y)**2)
        return mse

    def get_activ(self, func):
        if func == 'Relu':
            return self.Relu
        elif func == 'Leaky_Relu':
            return self.Leaky_Relu
        elif func == 'Sigmoid':
            return self.Sigmoid
        else:
            return self.Identity

    def get_grad(self, func):
        def Relu(x):
            return np.heaviside(x, 0)
        def Leaky_Relu(x):
            return np.heaviside(x>0, 0.2)
        def Sigmoid(x):
            return self.Sigmoid(x)*(1-self.Sigmoid(x))
        def Identity(x):
            return 1
        if func == 'Relu':
            return Relu
        elif func == 'Leaky_Relu':
            return Leaky_Relu
        elif func == 'Sigmoid':
            return Sigmoid
        else:
            return Identity

    def Relu(self, x):
        return np.maximum(x, 0)
    def Leaky_Relu(self, x):
        return np.maximum(0.2*x, x)
    def Sigmoid(self, x):
        return np.where(x>0, 1/(1+np.exp(-1*x)), np.exp(x)/(1+np.exp(x)))        
    def Identity(self, x):
        return x

nn = FCNN([1, 4, 4, 1], ["Identity", "Leaky_Relu","Leaky_Relu", "Identity"], 5000000, 0.5*(10**-7))

xs = np.c_[xs]
ys = np.c_[ys]

nn.training(xs,ys)
nn.plot_validerr()
nn.predict_and_plot(xs,ys)
print("w_s: ", nn.w_s)
print("b_s: ", nn.b_s)