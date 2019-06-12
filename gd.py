import numpy as np

class gd_1d:
    
    def __init__(self, fn_loss, fn_grad):
        self.fn_loss = fn_loss
        self.fn_grad = fn_grad
        
    def pv(self, x_init, n_iter, eta, tol):
        x = x_init
        
        loss_path = []
        x_path = []
        
        x_path.append(x)
        loss_this = self.fn_loss(x)
        loss_path.append(loss_this)
        g = self.fn_grad(x)

        for i in range(n_iter):
            if np.abs(g) < tol or np.isnan(g):
                break
            g = self.fn_grad(x)
            x += -eta * g
            x_path.append(x)
            loss_this = self.fn_loss(x)
            loss_path.append(loss_this)
            
        if np.isnan(g):
            print('Exploded')
        elif np.abs(g) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x = {}'.format(i, loss_this, x))
        self.loss_path = np.array(loss_path)
        self.x_path = np.array(x_path)
        
    def momentum(self, x_init, n_iter, eta, tol, alpha):
        x = x_init
        
        loss_path = []
        x_path = []
        
        x_path.append(x)
        loss_this = self.fn_loss(x)
        loss_path.append(loss_this)
        g = self.fn_grad(x)
        nu = 0

        for i in range(n_iter):
            g = self.fn_grad(x)
            if np.abs(g) < tol or np.isnan(g):
                break

            nu = alpha * nu + eta * g
            x += -nu
            x_path.append(x)
            loss_this = self.fn_loss(x)
            loss_path.append(loss_this)

        if np.isnan(g):
            print('Exploded')
        elif np.abs(g) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x = {}'.format(i, loss_this, x))
        self.loss_path = np.array(loss_path)
        self.x_path = np.array(x_path)

    def nag(self, x_init, n_iter, eta, tol, alpha):
        x = x_init
        
        loss_path = []
        x_path = []
        
        x_path.append(x)
        loss_this = self.fn_loss(x)
        loss_path.append(loss_this)
        g = self.fn_grad(x)
        nu = 0

        for i in range(n_iter):
            # i starts from 0 so add 1
            # The formula for mu was mentioned by David Barber UCL as being Nesterovs suggestion
            mu = 1 - 3 / (i + 1 + 5) 
            g = self.fn_grad(x - mu*nu)
            if np.abs(g) < tol or np.isnan(g):
                break

            nu = alpha * nu + eta * g
            x += -nu
            x_path.append(x)
            loss_this = self.fn_loss(x)
            loss_path.append(loss_this)

        if np.isnan(g):
            print('Exploded')
        elif np.abs(g) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x = {}'.format(i, loss_this, x))
        self.loss_path = np.array(loss_path)
        self.x_path = np.array(x_path)
        