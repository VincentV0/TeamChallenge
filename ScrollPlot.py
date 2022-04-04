from matplotlib.backend_bases import MouseButton
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, ax, X, y=None, markers=None, true_markers=None, ax_title="Image Viewer", close_on_click=False):
        
        # Initialize variables
        self.ax = ax
        ax.set_title(ax_title)
        self.X = X
        self.y = y
        self.marked_points = []
        self.markers = markers
        self.true_markers = true_markers
        self.markerplot = dict()
        self.scroll = 1
        self.close_on_click = close_on_click
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        # Show the image and mask
        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        if self.y is not None:
            self.mask = self.ax.imshow(self.y[:, :, self.ind], cmap='jet', alpha=0.4)

        # Show the markers
        self.plot_markers()
        self.manual_marks, = self.ax.plot([])
        self.true_marks, = self.ax.plot([])
        self.update()

    def on_scroll(self, event):
        # Scroll up and down
        if event.button == 'up':
            self.ind = (self.ind + self.scroll) % self.slices
        else:
            self.ind = (self.ind - self.scroll) % self.slices
        self.update()

    def on_click(self, event):
        # Left mouse button: select a point
        if event.button is MouseButton.LEFT:
            if event.xdata is not None and event.ydata is not None:
                x_marked, y_marked = round(event.xdata), round(event.ydata)
                self.marked_points.append([x_marked, y_marked, self.ind])
                self.op_slice_number = self.ind % self.slices
                self.update()

        # Right mouse button: increase scroll speed
        elif event.button is MouseButton.RIGHT:
            if self.scroll == 1:
                self.scroll = 3
            elif self.scroll == 3:
                self.scroll = 1
            else:
                self.scroll = 1
        else:
            print("Unknown button press")

        # Only one click allowed? Close plot
        if self.close_on_click:
            plt.close()

    def plot_markers(self):
        """
        Plots the landmarks in the ScrollPlot. Uses the color-dictionary defined as a constant at the start of this file.
        """       
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
        """
        Plots the manually placed markers (selected by the user)
        """
        self.manual_marks.set_xdata([])
        self.manual_marks.set_ydata([])

        mrks = np.array(self.marked_points)
        if len(mrks>0):
            x = mrks[mrks[:,2]==self.ind][:,0]
            y = mrks[mrks[:,2]==self.ind][:,1]
            self.manual_marks, = self.ax.plot(x, y, marker='x', markersize=3, color='w', linestyle='')
            self.manual_marks.axes.figure.canvas.draw()

    def plot_true_marks(self):
        """
        Plots the manually placed markers (derived from the xml-files)
        """
        self.true_marks.set_xdata([])
        self.true_marks.set_ydata([])

        x = self.true_markers[self.true_markers[:,2]==self.ind][:,0]
        y = self.true_markers[self.true_markers[:,2]==self.ind][:,1]
        #self.ax.set_color_cycle(['b', 'g', 'r', 'c', 'm', 'y', '#d4f542', '#a742f5'])
        self.true_marks, = self.ax.plot(x, y, marker=f'P', markersize=5, color='w', linestyle='')
        self.true_marks.axes.figure.canvas.draw()

    def update(self):
        """
        Update the image and its markers/masks
        """
        self.im.set_data(self.X[:, :, self.ind])
        if self.y is not None:
            self.mask.set_data(self.y[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        if self.y is not None:
            self.mask.axes.figure.canvas.draw()
        self.plot_markers()
        self.plot_manual_marks()
        if self.true_markers is not None:
            self.plot_true_marks()

    
    def get_marked_points(self):
        # Return the points marked by the user
        return np.array(self.marked_points)

