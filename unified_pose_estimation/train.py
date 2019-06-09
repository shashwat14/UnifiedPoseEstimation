import random

from tqdm import tqdm
import torch


from cfg import parameters
from net import UnifiedNetwork
from dataset import UnifiedPoseDataset

from tensorboardX import SummaryWriter

training_dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train2')
testing_dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test2')
samples = []
samples += training_dataset.samples + testing_dataset.samples
print samples[0]
random.shuffle(samples)
print samples[0]
training_dataset.samples = samples[:len(samples)/2]
testing_dataset.samples = samples[len(samples)/2:]

training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = 16, shuffle=True, num_workers=4)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size = 16, shuffle=False, num_workers=4)

model = UnifiedNetwork()
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)

best_loss = float('inf')

writer = SummaryWriter()

for epoch in range(parameters.epochs):
    
    # train
    
    model.train()
    training_loss = 0.
    for batch, data in enumerate(tqdm(training_dataloader)):

        optimizer.zero_grad()

        image = data[0]
        true = [x.cuda() for x in data[1:]]

        pred = model(image.cuda())
        loss = model.total_loss(pred, true)
        training_loss += loss.data.cpu().numpy()  
        loss.backward()

        optimizer.step()
    
    training_loss = training_loss / batch
    writer.add_scalar('error/total_training_error', training_loss, epoch)

    # validation
    #model.eval()
    validation_loss = 0.
    with torch.no_grad():
        for batch, data in enumerate(tqdm(testing_dataloader)):

            image = data[0]
            true = [x.cuda() for x in data[1:]]

            pred = model(image.cuda())
            loss = model.total_loss(pred, true)
            validation_loss += loss.data.cpu().numpy()

    validation_loss = validation_loss / batch
    writer.add_scalar('error/total_validation_error', validation_loss, epoch)

    if validation_loss < best_loss:

        print ("Old loss: {}, New loss : {}. Saving model to disk.".format(best_loss, validation_loss))
        best_loss = validation_loss

        torch.save(model.state_dict(), '../models/unified_net.pth')
    
    print ("Epoch : {} finished. Training Loss: {}. Validation Loss: {}".format(epoch, training_loss, validation_loss))