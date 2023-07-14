import torch

class LeNet(torch.nn.Module):
    """Implements LeNet without activation function

    """    
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding = 2
        )

        self.subsampling1 = torch.nn.MaxPool2d(
            kernel_size=2
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )

        self.subsampling2 = torch.nn.MaxPool2d(
            kernel_size= 2,
        )

        self.fc1 = torch.nn.Linear(
            in_features=16 * 5*5,
            out_features= 120
        )

        self.fc2 = torch.nn.Linear(
            in_features=120,
            out_features= 84
        )
        
        self.fc3 = torch.nn.Linear(
            in_features=84,
            out_features= 10
        )

    def forward(self, x : torch.Tensor):
        x = self.conv1(x)
        # print(f"Shape after first conv: {x.shape}")
        x = self.subsampling1(x)
        # print(f"Shape after first subsampling: {x.shape}")
        x = self.conv2(x)
        # print(f"Shape after second conv: {x.shape}")
        x = self.subsampling2(x)
        # print(f"Shape after second subsampling: {x.shape}")
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output