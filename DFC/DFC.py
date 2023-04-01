import numpy as np
import torch

device = "mps"
# Implementation based on https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip


# Configure
args = {
    "scribble": False,
    "nChannel": 10,
    "maxIter": 50,
    "minLabels": 0,
    "lr": 0.1,
    "nConv": 3,
    "visualize": 1,
    "input": None,
    "stepsize_sim": 1,
    "stepsize_con": 1,
    "stepsize_scr": 0.5
}

# Define model architecture
class MyNet(torch.nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_dim, args["nChannel"], kernel_size=3, stride=1, padding=1 )
        self.bn1 = torch.nn.BatchNorm2d(args["nChannel"])
        self.conv2 = torch.nn.ModuleList()
        self.bn2 = torch.nn.ModuleList()
        for i in range(args["nConv"]-1):
            self.conv2.append( torch.nn.Conv2d(args["nChannel"], args["nChannel"], kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( torch.nn.BatchNorm2d(args["nChannel"]) )
        self.conv3 = torch.nn.Conv2d(args["nChannel"], args["nChannel"], kernel_size=1, stride=1, padding=0 )
        self.bn3 = torch.nn.BatchNorm2d(args["nChannel"])

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu( x )
        x = self.bn1(x)
        for i in range(args["nConv"]-1):
            x = self.conv2[i](x)
            x = torch.nn.functional.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# Define segmentation function
def dfc_segment(im):
    # Load image
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) ).to(device)

    # Instantiate model
    model = MyNet(data.size(1)).to(device)
    model.train()

    # Define loss functions
    loss_fn = torch.nn.CrossEntropyLoss()
    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(size_average = True).to(device)
    loss_hpz = torch.nn.L1Loss(size_average = True).to(device)
    HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], args["nChannel"]).to(device)
    HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, args["nChannel"]).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=0.9)

    # Train on image
    for i in range(args["maxIter"]):
        optimizer.zero_grad()

        # forward pass
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, args["nChannel"] )

        # calculate continuity loss
        outputHP = output.reshape( (im.shape[0], im.shape[1], args["nChannel"]) )
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy.to(device),HPy_target.to(device))
        lhpz = loss_hpz(HPz.to(device),HPz_target.to(device))

        # calculate similarity loss
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        loss = args["stepsize_sim"] * loss_fn(output, target) + args["stepsize_con"] * (lhpy + lhpz)

        # backpropagate
        loss.backward()
        optimizer.step()
        
        # terminate if the number of labels is less than minLabels
        if nLabels <= args["minLabels"]:
            print ("nLabels", nLabels, "reached minLabels", args["minLabels"], ".")
            break
    
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args["nChannel"] )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()

    return im,target.data.cpu().numpy().reshape( im.shape[:2] ).astype( np.uint8 )