import torch
import torch.nn as nn
import json

config = json.load('config.json')

class YOLO_v1(nn.Module):
    '''
    YOLO base model 
    '''
    def __init__(self):
        super().__init__()

        self.final_depth = config['B'] * 5 + config['C']

        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, 7, 2, 'same'),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, 1, 'same'),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 128, 1, 1, 'same'),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, 3, 1, 'same'),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, 1, 1, 'same'),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, 3, 1, 'same'),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2)
        ])
        for i in range(4):
            self.layers.append([
                nn.Conv2d(512, 256, 1, 1, 'same'), 
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(256, 512, 3, 1, 'same'),
                nn.LeakyReLU(negative_slope = 0.1),
            ])

        self.layers.append([
            nn.Conv2d(512, 512, 1, 1, 'same'),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(512, 1024, 3, 1, 'same'),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.MaxPool2d(2, 2)
        ])

        for i in range(2):
            self.layers.append([nn.Conv2d(1024, 512, 1, 1, 'same'), 
                                nn.LeakyReLU(negative_slope = 0.1),
                                nn.Conv2d(512, 1024, 3, 1, 'same'),
                                nn.LeakyReLU(negative_slope = 0.1)
                                ])
        self.layers.append([
            nn.Conv2d(1024, 1024, 3, 1, 'same'),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(1024, 1024, 3, 2, 'same'),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(1024, 1024, 3, 1, 'same'),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(1024, 1024, 3, 1, 'same'),
            nn.LeakyReLU(negative_slope = 0.1),
        ]
        )

        self.layers.append([
            nn.Flatten(),
            nn.Linear(config['S']*config['S']*1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(4096, config['S']*config['S']*self.final_depth)
        ])

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return torch.reshape(self.model(x), (x.shape[0], config.S, config.S, -1))

