from tqdm import tqdm
import torch

import numpy as np

from cfg import parameters
from net import UnifiedNetwork
from dataset import UnifiedPoseDataset

testing_dataset = UnifiedPoseDataset(mode='test', loadit=True)

testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size = 1, shuffle=False, num_workers=1)

model = UnifiedNetwork()
model.load_state_dict(torch.load('../models/unified_net.pth'))
model.eval()
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)
    
# validation

with torch.no_grad():

    for batch, data in enumerate(tqdm(testing_dataloader)):

        image = data[0]
        true = [x.cuda() for x in data[1:]]

        pred = model(image.cuda())
        loss = model.total_loss(pred, true)
        
        print loss.data.cpu().numpy()

        hand_pose, action_prob, hand_conf, object_pose, object_prob, object_conf = pred
        object_conf = object_conf.squeeze(0)
        object_conf = object_conf.data.cpu().numpy()
        print np.unravel_index(object_conf.argmax(), object_conf.shape)
        break