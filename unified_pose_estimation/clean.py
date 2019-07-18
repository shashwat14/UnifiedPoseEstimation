from dataset import UnifiedPoseDataset

dataset = UnifiedPoseDataset(mode='train', loadit=False, name='train')
print len(dataset)
dataset = UnifiedPoseDataset(mode='test', loadit=False, name='test')
print len(dataset)

