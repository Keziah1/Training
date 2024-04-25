class SVM:
    def __init__(self, C=1, lr=0.15, max_iter=500):
        self.w = None
        self.b = None
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
    
    def loss(self, X, Y):
        regularizer = 0.5 * np.dot(self.w, self.w)
        liste_somme = []
        N = X.shape[0]
        
        #Calcul de la somme
        for i in range(N):
            valeur = 1 - Y[i]*(self.w.T*X[i,:]+self.b)
            liste_somme.append(valeur)
        somme = 0
        for i in range(N):
            if liste_somme[i]>0:
                somme += liste_somme[i]
                
        error_term = self.C*somme # you may need two conditions.
        loss = regularizer + error_term
        return loss
    
    def derivative_loss(self, xi, yi):
        """
        Parameters:
        ------------
        xi: (d) array, a data sample
        yi: int, lable of xi
        C: float, slackness for error term
        w:
        b:
    
        Return:
        -------------
        d_w:
        d_b:
        """
        
        d_w = self.w.T
        d_b = 0
        
        if 1 - yi * (np.dot(self.w.T,xi) + self.b) > 0:
            d_w += -self.C*yi*xi
            d_b = -self.C*yi
        
        return d_w, d_b
    
    def fit_SGD(self, X, Y):
        d = X.shape[1]
        self.w = np.ones(d)
        self.b = 1
        
        for epoch in range(max_iter):
            for xi, yi in zip(X, Y):
                d_w, d_b = self.derivative_loss(xi, yi)
    
                self.w = self.w - self.lr * d_w
                self.b = self.b - self.lr * d_b
    
        return self.w, self.b
    
    
    def predict(self, X):
        """
        Return the predicted label of all data in X
        """
        liste = []
        for xi in X:
            i = 0
            value = 1 - (np.dot(self.w.T, xi) + self.b)
            if value > 0:
                liste.append(1)
            else:
                liste.append(-1)
            i += 1
            
        return liste