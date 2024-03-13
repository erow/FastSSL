"""create puzzle problem for detecting feature suppression. """
from enum import Enum
import numpy as np
import torch
from torch.utils.data import IterableDataset,Dataset
import torchvision
import cv2
from PIL import Image


class PuzzleType(Enum):
    Independent=1
    Dependent = 2

def generate_puzzle(color,shape):
    radius = np.random.randint(4, 22)
    pos = (np.random.randint(0+radius, 64-radius), np.random.randint(0+radius, 64-radius))
    image = np.zeros((64, 64, 3), np.uint8)
    if shape == 0:
        # +
        image = cv2.line(image, (pos[0]-radius, pos[1]), (pos[0]+radius, pos[1]), color, thickness=2)
        image = cv2.line(image, (pos[0], pos[1]-radius), (pos[0], pos[1]+radius), color, thickness=2)
    elif shape == 1:
        pt1 = (pos[0], pos[1] - radius)
        pt2 = (pos[0] - radius, pos[1] + radius)
        pt3 = (pos[0] + radius, pos[1] + radius)
        triangle_cnt = np.array([pt1, pt2, pt3])
        image = cv2.drawContours(image, [triangle_cnt], 0, color, -1)
    elif shape == 2:
        image = cv2.ellipse(image, pos, (radius, radius), 0, 0, 180, color, -1)
    elif shape == 3:
        image = cv2.rectangle(image, (pos[0]-radius//4, pos[1]-radius), (pos[0]+radius//4, pos[1]+radius), color, -1)
    elif shape == 4:
        image = cv2.circle(image, pos, radius, color, -1) 
    else:
        # If the shape is not recognized, return a blank image
        pass
    
    image = Image.fromarray(image)
    return image

class Puzzle(Dataset):
    def __init__(self, puzzle_type, num_samples, transform=None):
        self.puzzle_type = puzzle_type
        self.num_samples = num_samples
        self.transform = transform
        self.colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (0, 255, 0)]

    def __getitem__(self, index):
        factor_size=5
        if self.puzzle_type == PuzzleType.Independent:
            shape = np.random.randint(0, factor_size)
            c = np.random.randint(0, factor_size)
        else:
            shape = np.random.randint(0, factor_size)
            c = shape
        
        color = self.colors[c]
        image = generate_puzzle(color,shape)
        if self.transform:
            image = self.transform(image)
        return image,(color,shape)
    
    def __len__(self):
        return self.num_samples  

    def _generate_puzzle(self):
        for _ in range(self.num_samples):
            image = self[0]
            yield image
    def __iter__(self):
        return self._generate_puzzle()

if __name__ == '__main__':
    puzzle = Puzzle(PuzzleType.Dependent, 16)
    images = np.array(list(iter(puzzle)))
    images = torch.from_numpy(images)
    torchvision.utils.save_image(images.float(),"puzzle.png", nrow=4,normalize=True)
