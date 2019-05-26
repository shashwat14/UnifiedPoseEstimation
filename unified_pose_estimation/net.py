import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.action_ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.object_ce_loss = nn.CrossEntropyLoss(reduction='none')

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
        pred_hand_pose = pred_v_h[:, :3*self.num_hand_control_points, :, :, :] # TODO: root control point must have sigmoid activation
        pred_action_prob = pred_v_h[:, 3*self.num_hand_control_points:-1, :, :, :] 
        pred_hand_conf = F.sigmoid(pred_v_h[:, -1, :, :, :])

        # object specific predictions
        pred_object_pose = pred_v_o[:, :3*self.num_object_control_points, :, :, :] # TODO: root control point must have sigmoid activation
        pred_object_prob = pred_v_o[:, 3*self.num_object_control_points:-1, :, :, :]
        pred_object_conf = F.sigmoid(pred_v_o[:, -1, :, :, :])

        return pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf
        
    def total_loss(self, pred, true):

        pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = pred
        true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = true
        
        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, hand_mask) + self.pose_loss(pred_object_pose, true_object_pose, object_mask)
        total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask) + self.conf_loss(pred_object_conf, pred_object_pose, true_hand_pose, object_mask)
        total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)

        total_loss = total_pose_loss + total_conf_loss + total_action_loss + total_object_loss

        return total_loss

    def pose_loss(self, pred, true, mask):

        pred = pred.view(32, 21, 3, 5, 13, 13)
        true = true.view(32, 21, 3, 5, 13, 13)
        masked_pose_loss = torch.mean(torch.sum(mask * torch.sum(torch.mul(pred - true, pred - true), dim=[1,2]), dim=[1,2,3]))
        return masked_pose_loss
    
    def conf_loss(self, pred_conf, pred, true, mask):
        
        pred = pred.view(32, 21, 3, 5, 13, 13)
        true = true.view(32, 21, 3, 5, 13, 13)

        pred_pixel = pred[:, :, :2, :, :, :]
        pred_depth = pred[:, :, 2, :, :, :]

        true_pixel = true[:, :, :2, :, :, :]
        true_depth = true[:, :, 2, :, :, :]

        pixel_distance = torch.sqrt(torch.sum(torch.mul(pred_pixel - true_pixel, pred_pixel - true_pixel), dim=2))
        depth_distance = torch.sqrt(torch.mul(pred_depth - true_depth, pred_depth - true_depth))
        
        # threshold
        pixel_distance_mask = (pixel_distance < parameters.pixel_threshold).type(torch.FloatTensor)
        depth_distance_mask = (depth_distance < parameters.depth_threshold).type(torch.FloatTensor)

        pixel_distance = torch.mean(pixel_distance_mask * pixel_distance, dim=1)
        depth_distance = torch.mean(depth_distance_mask * depth_distance, dim=1)

        pixel_conf = torch.exp(parameters.sharpness * (1 - pixel_distance / parameters.pixel_threshold)) / torch.exp(parameters.sharpness * (1 - torch.zeros(pixel_distance.size())))
        depth_conf = torch.exp(parameters.sharpness * (1 - depth_distance / parameters.depth_threshold)) / torch.exp(parameters.sharpness * (1 - torch.zeros(depth_distance.size())))
        true_conf = 0.5 * (pixel_conf + depth_conf)
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        exist_conf_error = torch.mean(torch.sum(mask * squared_conf_error, dim=[1,2,3]))

        true_conf = torch.zeros(pred_conf.size())
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        no_exist_conf_error = torch.mean(torch.sum((1 - mask) * squared_conf_error, dim=[1,2,3]))

        return 5 * exist_conf_error + 0.1 * no_exist_conf_error
        
    def action_loss(self, pred, true, mask):
        action_ce_loss = self.action_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * action_ce_loss, dim=[1,2,3]))

    def object_loss(self, pred, true, mask):
        object_ce_loss = self.object_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * object_ce_loss, dim=[1,2,3]))

if __name__ == '__main__':

    model = UnifiedNetwork()
    x = torch.randn(32, 3, 416, 416)

    true = torch.randn(32, 76, 5, 13, 13), torch.randn(32, 74, 5, 13, 13)

    pred =  model(x)
    
    true_hand_pose = torch.randn(32, 3 * parameters.num_hand_control_points, 5, 13, 13)
    true_action_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_actions)
    hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
    hand_mask[0, 0, 0] = 1.

    true_object_pose = torch.randn(32, 3 * parameters.num_object_control_points, 5, 13, 13)
    true_object_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_objects)
    object_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
    object_mask[0, 0, 0] = 1.

    true = true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask

    print model.total_loss(pred, true)
    