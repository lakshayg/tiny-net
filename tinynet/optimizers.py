import numpy as np

def sgd(lr=1e-4, mu=0.9, weight_decay=0):
    def aux_init(param):
        return np.zeros_like(param)
    def update_rule(x, dx, aux):
        x = (1-weight_decay) * x
        aux = mu * aux - lr * dx
        x += aux
        return x, aux
    return aux_init, update_rule

def nag(lr=1e-4, mu=0.9, weight_decay=0):
    def aux_init(param):
        return np.zeros_like(param)
    def update_rule(x, dx, aux):
        x = (1-weight_decay) * x
        v_prev = np.copy(aux)
        aux = mu * aux - lr * dx
        x += -mu * v_prev + (1+mu) * aux
        return x, aux
    return aux_init, update_rule

def adagrad(lr=1e-4, eps=1e-8, weight_decay=0):
    def aux_init(param):
        return np.zeros_like(param)
    def update_rule(x, dx, aux):
        x = (1-weight_decay) * x
        aux += dx**2
        x += -lr * dx / (np.sqrt(aux) + eps)
        return x, aux
    return aux_init, update_rule

def rmsprop(lr=1e-4, decay_rate=0.9, eps=1e-8, weight_decay=0):
    def aux_init(param):
        return np.zeros_like(param)
    def update_rule(x, dx, aux):
        x = (1-weight_decay) * x
        aux = (1-decay_rate) * dx**2 + decay_rate * aux
        x += -lr * dx / (np.sqrt(aux) + eps)
        return x, aux
    return aux_init, update_rule

def adam(lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
    def aux_init(param):
        return [np.zeros_like(param)]*2 # m, v
    def update_rule(x, dx, aux):
        x = (1-weight_decay) * x
        aux[0] = beta1 * aux[0] + (1-beta1) * dx
        aux[1] = beta2 * aux[1] + (1-beta2) * (dx**2)
        x += -lr * aux[0] / (np.sqrt(aux[1]) + eps)
        return x, aux
    return aux_init, update_rule

