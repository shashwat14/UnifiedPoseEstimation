import torch
import torch.nn as nn

import torchvision.models as models

from cfg import parameters

class UnifiedNetwork(nn.Module):

    def __init__(self):

        super(UnifiedNetwork, self).__init__()

        self.num_hand_control_points = parameters.num_hand_control_points
        self.num_object_control_points = parameters.num_object_control_points
        self.num_actions = parameters.num_actions
        self.num_objects = parameters.num_objects
        self.depth_discretization = parameters.depth_discretization
        
        model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-2])

        self.hand_vector_size = 3 * self.num_hand_control_points + 1 + self.num_actions
        self.object_vector_size = 3 * self.num_hand_control_points + 1 + self.num_objects
        self.target_channel_size = self.depth_discretization * ( self.hand_vector_size + self.object_vector_size )

        # prediction layers
        self.conv = nn.Conv2d(512, self.target_channel_size, (3,3), padding=1)

        # losses
        self.setup_losses()
    
    def setup_losses(self):

        self.pose_loss = nn.MSELoss()
        self.conf_loss = nn.MSELoss()
        self.action_loss = nn.CrossEntropyLoss()
        self.object_loss = nn.CrossEntropyLoss()

    def forward(self, x):

        # split it into different types of data
        height, width = x.size()[2:]
        
        assert height == width
        assert height % 32 == 0
        
        target_height, target_width = height / 32, width / 32

        x = self.features(x)
        x = self.conv(x).view(-1, self.hand_vector_size + self.object_vector_size, self.depth_discretization, target_height, target_width)
        
        pred_v_h = x[:, :self.hand_vector_size, :, :, :]
        pred_v_o = x[:, self.hand_vector_size:, :, :, :]

        # hand specific predictions
        pred_hand_pose = pred_v_h[:, :3*self.num_hand_control_points, :, :, :]
        pred_action_prob = pred_v_h[:, 3*self.num_hand_control_points:-1, :, :, :]
        pred_hand_conf = pred_v_h[:, -1, :, :, :]

        # object specific predictions
        pred_object_pose = pred_v_o[:, :3*self.num_object_control_points, :, :, :]
        pred_object_prob = pred_v_o[:, 3*self.num_object_control_points:-1, :, :, :]
        pred_object_conf = pred_v_o[:, -1, :, :, :]

        return pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf
        
    def total_loss(self, pred, true):

        pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = pred
        true_hand_pose, true_action_prob, true_hand_conf, true_object_pose, true_object_prob, true_object_conf = true
        
        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose) + self.pose_loss(pred_object_pose, true_object_pose)
        total_conf_loss = self.conf_loss(pred_hand_conf, true_hand_conf) + self.conf_loss(pred_object_conf, true_object_conf)
        total_action_loss = self.action_loss(pred_action_prob, true_action_prob)
        total_object_loss = self.object_loss(pred_object_prob, true_object_prob)

        weighted_pose_loss = parameters.pose_loss_weight * total_pose_loss
        weighted_conf_loss = parameters.conf_loss_weight * total_conf_loss
        weighted_action_loss = parameters.action_loss_weight * total_action_loss
        weigthed_object_loss = parameters.object_loss_weight * total_object_loss

        total_loss = weighted_pose_loss + weighted_conf_loss + weighted_action_loss + weigthed_object_loss

        return total_loss

if __name__ == '__main__':

    model = UnifiedNetwork()
    x = torch.randn(32, 3, 416, 416)

    true = torch.randn(32, 76, 5, 13, 13), torch.randn(32, 74, 5, 13, 13)

    pred =  model(x)
    
    true_hand_pose = torch.randn(32, 3 * parameters.num_hand_control_points, 5, 13, 13)
    true_action_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_actions)
    true_hand_conf = torch.randn(32, 5, 13, 13)
    
    true_object_pose = torch.randn(32, 3 * parameters.num_object_control_points, 5, 13, 13)
    true_object_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_objects)
    true_object_conf = torch.randn(32, 5, 13, 13)

    true = true_hand_pose, true_action_prob, true_hand_conf, true_object_pose, true_object_prob, true_object_conf

    print model.total_loss(pred, true)
    