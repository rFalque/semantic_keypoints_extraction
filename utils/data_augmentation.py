import numpy as np
import random

class DataAugmentation:
    def __init__(self):
        # scale
        self.x_scale = None
        self.y_scale = None
        self.z_scale = None
        self.tform_scale = None

        # flip
        self.flip = None
        self.tform_flip = None
        
        # shear_forward
        self.forward_shearing = None
        self.tform_shear_forward = None

        # shear_sideway
        self.sideway_shearing = None
        self.tform_shear_sideway = None

        # translation
        self.x_offset = None
        self.y_offset = None
        self.z_offset = None
        self.tform_offset = None

        # final transform
        self.tform = None

    def transform_scale(self, x_scale=None, y_scale=None, z_scale=None):
        """Scale with respect to the x, y, and z axis. Default values for scaling is drawn from a Gaussian distribution N(1, 0,1)."""
        if x_scale is None:
            self.x_scale = random.gauss(1, 0.1)
        else:
            self.x_scale = x_scale
        if y_scale is None:
            self.y_scale = random.gauss(1, 0.1)
        else:
            self.y_scale = y_scale
        if z_scale is None:
            self.z_scale = random.gauss(1, 0.1)
        else:
            self.z_scale = z_scale
        self.tform_scale = np.identity(4)
        self.tform_scale[0, 0] = self.x_scale
        self.tform_scale[1, 1] = self.y_scale
        self.tform_scale[2, 2] = self.z_scale
        return self.tform_scale

    def transform_flip(self, flip=None):
        """Flip with the YZ axis of symmetry. Default value for flip is drawn from a uniform distribution between 0 and 1, call with flip=1 to force flipping."""
        if flip is None:
            self.flip = (random.random() > 0.5)
        else:
            self.flip = (flip > 0.5)
        self.tform_flip = np.identity(4)
        if (self.flip):
            self.tform_flip[0, 0] = -1
        return self.tform_flip
        
    def transform_shear_forward(self, shearing=None):
        """Shear to simulate leaning forward and backward. Default value for shearing is drawn from a Gaussian distribution N(0, 0,2)."""
        if shearing is None:
            self.shearing = random.gauss(0, 0.2)
        else:
            self.shearing = shearing
        self.tform_shear_forward = np.identity(4)
        self.tform_shear_forward[1, 2] = self.shearing
        return self.tform_shear_forward

    def transform_shear_sideway(self, shearing=None):
        """Shear to simulate sideway motion. Default value for shearing is drawn from a Gaussian distribution N(0, 0,2)."""
        if shearing is None:
            self.shearing = random.gauss(0, 0.2)
        else:
            self.shearing = shearing
        self.tform_shear_sideway = np.identity(4)
        self.tform_shear_sideway[1, 0] = self.shearing
        return self.tform_shear_sideway

    def transform_translate(self, x_offset=None, y_offset=None, z_offset=None):
        """offset with respect to the x, y, and z axis. Default values for scaling is drawn from a Gaussian distribution N(1, 0,1)."""
        if x_offset is None:
            self.x_offset = random.gauss(0, 0.1)
        else:
            self.x_offset = x_offset
        if y_offset is None:
            self.y_offset = random.gauss(0, 0.1)
        else:
            self.y_offset = y_offset
        if z_offset is None:
            self.z_offset = random.gauss(0, 0.1)
        else:
            self.z_offset = z_offset
        self.tform_offset = np.identity(4)
        self.tform_offset[0, 3] = self.x_offset
        self.tform_offset[1, 3] = self.y_offset
        self.tform_offset[2, 3] = self.z_offset
        return self.tform_offset

    def build_transform(self):
        """Transform for the data augmentation."""
        self.tform = np.identity(4)
        if (self.tform_scale is not None):
            self.tform = np.matmul(self.tform, self.tform_scale)
        if (self.tform_flip is not None):
            self.tform = np.matmul(self.tform, self.tform_flip)
        if (self.tform_shear_forward is not None):
            self.tform = np.matmul(self.tform, self.tform_shear_forward)
        if (self.tform_shear_sideway is not None):
            self.tform = np.matmul(self.tform, self.tform_shear_sideway)
        if (self.tform_offset is not None):
            self.tform = np.matmul(self.tform, self.tform_offset)
        return self.tform

    # for open3d structures
    def transform(self, shape):
        self.build_transform()
        return shape.transform(self.tform)
