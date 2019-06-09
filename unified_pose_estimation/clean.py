from dataset import UnifiedPoseDataset

dataset = UnifiedPoseDataset(mode='train', loadit=False, name='train2')
print len(dataset)
dataset = UnifiedPoseDataset(mode='test', loadit=False, name='test2')
print len(dataset)

