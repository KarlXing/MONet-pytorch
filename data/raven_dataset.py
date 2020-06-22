import os

import torchvision.transforms.functional as TF

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class RAVENDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=1, output_nc=1,
                            crop_size=160, # crop is done first
                            load_size=160,  # before resize
                            num_slots=9, display_ncols=9)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        self.A_paths = [path for path in self.A_paths if '.npz' in path]

    def _transform(self, img):
        img = torch.from_numpy(255-img).type(torch.FloatTensor)/255.0
        return img.unsqueeze(0) # channel as 1

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        file_index, img_index = divmod(index, 16)
        A_path = self.A_paths[file_index]
        A_img = np.load(A_path)['image'][img_index]
        A = self._transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)*16  # Each npz file is one question with 16 images
