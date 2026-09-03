import torch
import torch.nn as nn #nueral network package nn


class CarBrain(nn.Module):
    def __init__(self):
        super().__init__() 

        self.network = nn.Sequential( #run the network in sequential order
            nn.Linear(9, 16), #9 inputs (7 sensors + speed), 16 nuetrons
            nn.ReLU(),

            nn.Linear(16, 16),
            nn.ReLU(),

            nn.Linear(16, 3),
        )

    def forward(self, state):
        return self.network(state) #state = car state


model = CarBrain() #creates the model

print(model)