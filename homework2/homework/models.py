import torch


class CNNClassifier(torch.nn.Module):
    # block class for defining convolutional blocks
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            # Defines a sequence of layers as a part of the block. Two convolutional layers with ReLU activation functions in between
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.ReLU()
            )
        
        # Forward pass method for the Block class. Passing input x and passes it through the layers defined in self.net.
        def forward(self, x):
            return self.net(x)
        
    def __init__(self, layers=[32, 64, 128], n_input_channels=3, num_classes=6):
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        c = 32
        # Construct the CNN network using specified layers and input channels
        for layer in layers:
            # Appends a Block instance to the list L with the specified input channels (c), output channels (l), and a stride of 2.
            L.append(self.Block(c, layer, stride=2))
            c = layer
        # Define the network as a sequential layer
        self.network = torch.nn.Sequential(*L)
        
        # Global Average Pooling
        self.global_avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification
        self.classifier = torch.nn.Linear(c, num_classes)

    def forward(self, x):
        # Compute the features by passing input x through the network
        z = self.network(x)
        
        # Global Average Pooling
        z = self.global_avg_pooling(z)
        
        # Flatten the features
        z = z.view(z.size(0), -1)
        
        # Classify, Passes the pooled features through the linear classifier and returns the result.
        return self.classifier(z)
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        #raise NotImplementedError('CNNClassifier.forward')


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
