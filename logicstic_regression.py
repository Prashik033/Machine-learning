import numpy as np

class LogisticRegression:
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
        lr =  self.b_0 + self.b_1*x
        print(lr)
        sigmoid = 1/(1+np.exp(-lr))
        if sigmoid >= 0.5:
            print("pass")
        else:
            print("fail")
        return sigmoid
    
x = np.array([0.50,1.50,2.00,4.25,3.25,3.50],ndmin=2).reshape(6,1)
y = np.array([0,0,0,1,1,1])
model = LogisticRegression()
model.fit(x,y)
print(model.predict(2.70))