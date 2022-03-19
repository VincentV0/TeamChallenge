import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np

COLORS = {
    1: 'b',
    2: 'g',
    3: 'r',
    4: 'c',
    5: 'm',
    6: 'y',
    7: '#d4f542',
    8: '#a742f5'
}


class ScrollPlot:
    def __init__(self, ax, X, y=None, markers=None):
        self.ax = ax
        ax.set_title('Image Viewer')
        self.X = X
        self.y = y
        self.marked_points = []
        self.markers = markers
        self.markerplot = dict()

        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        if self.y is not None:
            self.mask = self.ax.imshow(self.y[:, :, self.ind], cmap='jet', alpha=0.4)

        self.plot_markers()
        self.manual_marks, = self.ax.plot([])
        self.update()

    def on_scroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            x_marked, y_marked = round(event.xdata), round(event.ydata)
            self.marked_points.append([x_marked, y_marked, self.ind])
            self.op_slice_number = self.ind % self.slices
            self.update()
        elif event.button is MouseButton.RIGHT:
            plt.close()
        else:
            print("Unknown button press")

    def plot_markers(self):
        if self.markers is not None:
            for marker_ID in self.markers:
                landmark = self.markers[marker_ID]
                if landmark[self.ind, 0] != -1 and landmark[self.ind, 1] != -1:
                    if marker_ID in self.markerplot:
                        self.markerplot[marker_ID].set_data(landmark[self.ind, 0], landmark[self.ind, 1])
                    else: 
                        self.markerplot[marker_ID], = self.ax.plot(landmark[self.ind, 0], landmark[self.ind, 1], marker='o', markersize=3, color=COLORS[marker_ID])
                else:
                    if marker_ID in self.markerplot:
                        self.markerplot[marker_ID].remove()
                        del self.markerplot[marker_ID]

    def plot_manual_marks(self):
        self.manual_marks.set_xdata([])
        self.manual_marks.set_ydata([])

        for marker in self.marked_points:
            if marker[2] == self.ind:
                self.manual_marks, = self.ax.plot(marker[0], marker[1], marker='x', markersize=3, color='w')
        self.manual_marks.axes.figure.canvas.draw()


    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        if self.y is not None:
            self.mask.set_data(self.y[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        if self.y is not None:
            self.mask.axes.figure.canvas.draw()
        self.plot_markers()
        self.plot_manual_marks()

            
    def get_marked_points(self):
        return np.array(self.marked_points)

