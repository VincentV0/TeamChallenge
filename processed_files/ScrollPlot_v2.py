import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

class ScrollPlot:

    def __init__(self, ax, X, y):
        self.ax = ax
        ax.set_title('Image Viewer')

        self.X = X
        self.y = y
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        self.x_marked = 0
        self.y_marked = 0
        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        self.mask = ax.imshow(self.y[:, :, self.ind], cmap='jet', alpha=0.4)
        self.update()

    def on_scroll(self, event):
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

    def on_click(self, event):
        
        if event.button is MouseButton.LEFT:
            self.x_marked, self.y_marked = round(event.xdata), round(event.ydata)
            self.op_slice_number = self.ind % self.slices
        else:
            print("Unknown button press")
            
    def get_landmark(self):
        return (self.x_marked, self.y_marked, self.ind)

