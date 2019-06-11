import numpy as np

class gd_pv_1d:
    
    def __init__(self, fn_loss, fn_grad):
        self.fn_loss = fn_loss
        self.fn_grad = fn_grad
        
    def find_min(self, x_init, n_iter, eta, tol):
        #self.x_init = x_init
        #self.n_iter = n_iter
        #self.eta = eta
        #self.tol = tol
        x = x_init
        
        loss_path = []
        x_path = []
        
        x_path.append(x)
        loss_this = self.fn_loss(x)
        loss_path.append(loss_this)
        g = self.fn_grad(x)

        for i in range(n_iter):
            if g < tol:
                break
            g = self.fn_grad(x)
            x += -eta * g
            x_path.append(x)
            loss_this = self.fn_loss(x)
            loss_path.append(loss_this)
            
        self.loss_path = loss_path
        self.x_path = x_path
        self.loss_fn_min = loss_this
        self.x_at_min = x