import matplotlib.pyplot as plt

class ScrollPlot:
    def __init__(self, ax, X, y):
        self.ax = ax
        ax.set_title('Image Viewer')


        self.X = X
        self.y = y
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        self.mask = ax.imshow(self.y[:, :, self.ind], cmap='jet', alpha=0.4)
        self.update()

    def on_scroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.mask.set_data(self.y[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        self.mask.axes.figure.canvas.draw()

