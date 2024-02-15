import numpy as np

class LinearRegression:
    def __init__(self):
        b_0 = 0
        b_1 = 0

    def fit(self,x,y):
        xm = np.mean(x)
        ym = np.mean(y)
        num = 0
        den = 0
        for i in range(len(x)):
            num += (x[i] - xm) * (y[i] - ym)
            den += (x[i] - xm)**2
        self.b_1 =  num/den
        self.b_0 = ym - (self.b_1*xm)
    
    def predict(self,x):
        return self.b_0 + self.b_1*x

x = np.array([160,171,182,180,154],ndmin=2)
x = x.reshape(5,1)
y = np.array([72,76,77,83,76])
model = LinearRegression()
model.fit(x,y)
print(model.predict(161))