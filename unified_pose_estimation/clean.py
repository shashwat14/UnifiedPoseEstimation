from dataset import UnifiedPoseDataset

dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train2')
print len(dataset)
dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test2')
print len(dataset)

