import numpy as np
class LogisticRegression:
    def __init__(self,epochs=500,lr=0.001):
        self.epochs=epochs
        self.lr=lr
        self.w=None
        self.b=None
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def log_loss(self,y_pred,y):
        m=len(y)
        return -(1/m) * (np.sum( y * np.log(y_pred) + (1-y) * np.log(1-y_pred)))
    def fit(self,X,y):
        m,n=X.shape
        self.w=np.zeros((n,1))
        self.b=0

        for i in range(self.epochs):
            z= np.dot(X,self.w) + self.b
            y_pred=self.sigmoid(z)
            loss=self.log_loss(y_pred,y)
            if(i %100==0):
                print(f'Epochs : {i+1} loss : {loss}')

            dw= (1/m) * (np.dot(X.T,(y_pred-y)))
            db= (1/m) * (np.sum(y_pred-y))
            self.w-= self.lr*dw
            self.b-= self.lr*db
    
    def predict(self,X):
        z=np.dot(X,self.w)+self.b
        y_pred=self.sigmoid(z)
        return [1 if i>0.5 else 0 for i in y_pred]
