import numpy as np

## Neural net layers
class Sequential:

    def __init__(self, layers=None):
        if type(layers) in [list, tuple]:
            self.layers = layers
        elif type(layers) == str:
            import dill
            self.layers = dill.load(open(layers, 'rb'))
        else:
            self.layers = None

    def configure(self, batch_size, optimizer, objective):
        for layer in self.layers:
            layer.configure(batch_size=batch_size, optimizer=optimizer)
        self.batch_size = batch_size
        self.loss = objective

    def saveas(self, filename):
        import dill
        dill.dump(self.layers, open(filename, 'wb'))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, de_dy):
        for layer in self.layers[::-1]:
            de_dy = layer.backward(de_dy)
        return de_dy

    def predict(self, x):
        for layer in self.layers:
            x = layer.predict(x)
        return x

    def update(self):
        for layer in self.layers:
            layer.update()

    def train(self, x, y, epochs, display=True):
        bs = self.batch_size
        lim = x.shape[0] + (x.shape[0]%bs == 0)*bs
        try:
            for i in range(epochs):
                for j in range(bs, lim, bs):
                    yp = self.forward(x[j-bs:j])
                    loss, deriv = self.loss(y[j-bs:j], yp)
                    if display:
                        print('loss: {}'.format(loss.mean()))
                    self.backward(deriv)
                    self.update()
        except KeyboardInterrupt:
            return

