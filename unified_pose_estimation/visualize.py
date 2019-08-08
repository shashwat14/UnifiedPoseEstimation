import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class UnifiedVisualization:

    
    def __init__(self):

        self.fig = plt.figure(figsize=(12,6))
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        
        self.ax.set_xlim((-50,250))
        self.ax.set_ylim((-50,250))
        self.ax.set_zlim((250,550))
        
    def plot_box(self, points):

        xs = points[:,0]
        ys = points[:,1]
        zs = points[:,2]
        self.ax.scatter3D(xs, ys, zs)    

        for i, (x, y, z) in enumerate(points):
            self.ax.text(x, y, z, str(i), color='red')
        
        # lines (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)
        # list of sides' polygons of figure
        verts = [[points[0],points[1],points[2],points[3]],
        [points[4],points[5],points[6],points[7]], 
        [points[0],points[1],points[5],points[4]], 
        [points[2],points[3],points[7],points[6]], 
        [points[1],points[2],points[6],points[5]],
        [points[4],points[7],points[3],points[0]]]

        # plot sides
        self.ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        

    def plot_hand(self, points):

        thumb = points[1:5]
        index = points[5:9]
        middle = points[9:13]
        ring = points[13:17]
        pinky = points[17:21]

        thumb_xs = thumb[:,0]
        thumb_ys = thumb[:,1]
        thumb_zs = thumb[:,2]
        self.ax.plot(thumb_xs, thumb_ys, thumb_zs)

        index_xs = index[:,0]
        index_ys = index[:,1]
        index_zs = index[:,2]
        self.ax.plot(index_xs, index_ys, index_zs)

        middle_xs = middle[:,0]
        middle_ys = middle[:,1]
        middle_zs = middle[:,2]
        self.ax.plot(middle_xs, middle_ys, middle_zs)

        ring_xs = ring[:,0]
        ring_ys = ring[:,1]
        ring_zs = ring[:,2]
        self.ax.plot(ring_xs, ring_ys, ring_zs)

        pinky_xs = pinky[:,0]
        pinky_ys = pinky[:,1]
        pinky_zs = pinky[:,2]
        self.ax.plot(pinky_xs, pinky_ys, pinky_zs)

        wrist_x = points[0,0]
        wrist_y = points[0,1]
        wrist_z = points[0,2]

        thumb_x = points[1,0]
        thumb_y = points[1,1]
        thumb_z = points[1,2]

        self.ax.plot([wrist_x, thumb_x], [wrist_y, thumb_y], [wrist_z, thumb_z])

        index_x = points[5,0]
        index_y = points[5,1]
        index_z = points[5,2]

        self.ax.plot([wrist_x, index_x], [wrist_y, index_y], [wrist_z, index_z])

        middle_x = points[9,0]
        middle_y = points[9,1]
        middle_z = points[9,2]

        self.ax.plot([wrist_x, middle_x], [wrist_y, middle_y], [wrist_z, middle_z])

        ring_x = points[13,0]
        ring_y = points[13,1]
        ring_z = points[13,2]

        self.ax.plot([wrist_x, ring_x], [wrist_y, ring_y], [wrist_z, ring_z])

        pinky_x = points[17,0]
        pinky_y = points[17,1]
        pinky_z = points[17,2]

        self.ax.plot([wrist_x, pinky_x], [wrist_y, pinky_y], [wrist_z, pinky_z])

    def plot_rgb(self, rgb):
        self.ax = self.fig.add_subplot(122)
        self.ax.imshow(rgb)

    def plot(self):

        plt.show()
    