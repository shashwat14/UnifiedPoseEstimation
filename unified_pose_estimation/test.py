from tqdm import tqdm
import torch

import numpy as np

from cfg import parameters
from net import UnifiedNetwork
from dataset import UnifiedPoseDataset
from visualize import UnifiedVisualization

training_dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test2')
training_dataset[0]
print training_dataset.samples[0]
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = 1, shuffle=False, num_workers=1)

model = UnifiedNetwork()
model.load_state_dict(torch.load('../models/unified_net.pth'))
model.eval()
model.cuda()

#optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)
    
# validation

with torch.no_grad():

    hand_cell_counter = 0.
    object_cell_counter = 0.
    object_counter = 0.
    action_counter = 0.

    for batch, data in enumerate(tqdm(training_dataloader)):
    
        image = data[0]
        true = [x.cuda() for x in data[1:]]

        pred = model(image.cuda())
        loss = model.total_loss(pred, true)
        
        #print loss.data.cpu().numpy()

        pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = [p.data.cpu().numpy() for p in pred]
        true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = [t.data.cpu().numpy() for t in true]

        true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
        true_object_cell = np.unravel_index(object_mask.argmax(), object_mask.shape)

        pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)
        pred_object_cell = np.unravel_index(pred_object_conf.argmax(), pred_object_conf.shape)
        
        hand_cell_counter += int(true_hand_cell == pred_hand_cell)
        object_cell_counter += int(true_object_cell == pred_object_cell)
        
        print hand_cell_counter, object_cell_counter
        z, v, u = true_hand_cell[1:]
        dels = pred_hand_pose[0,:,z, v, u].reshape(21, 3)
        del_u, del_v, del_z = dels[:,0], dels[:,1], dels[:,2]
        hand_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))

        z, v, u = true_object_cell[1:]
        dels = pred_object_pose[0,:,z, v, u].reshape(21, 3)
        del_u, del_v, del_z = dels[:,0], dels[:,1], dels[:,2]
        object_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))


        # print training_dataset.samples[batch]

        """
        viz = UnifiedVisualization()
        viz.plot_hand(hand_points)
        viz.plot_box(object_points[1:9, :])
        viz.plot()

        """
    print hand_cell_counter * 1. / batch
    print object_cell_counter * 1. / batch