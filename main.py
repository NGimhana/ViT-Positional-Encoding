from ViT16 import ViT16
from data_preprocessor import get_data_loader 
import torch
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataloader = get_data_loader("train", batch_size=4, shuffle=True)
val_dataloader = get_data_loader("val", batch_size=4, shuffle=False)
test_dataloader = get_data_loader("test", batch_size=4, shuffle=False)


model = ViT16(num_classes=10)
criterian = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.to(device)

def train(epoch_number):    
    running_loss = 0.
    last_loss = 0.
    
    for i, data in enumerate(train_dataloader):
        pixel_values, labels = data['pixel_values'], data['labels']
        pixel_values, labels = pixel_values.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # outputs
        logits = model(pixel_values)
        
        loss = criterian(logits, labels)
        
        ## compute gradients
        loss.backward()
        ## update weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
            
    writer.add_scalar('Loss/train', last_loss, epoch_number)
    return last_loss

def train_step():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    epoch_number = 0
    EPOCHS = 10
    best_vloss = 1_000_000.

    ## Training loop
    for epoch in range(EPOCHS):

        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        writer.flush()
        
        avg_loss = train(epoch)

        # We don't need gradients on to do reporting
        model.eval()
        
        with torch.no_grad():
            running_vloss = 0.
            correct_count = 0.
            for i, vdata in enumerate(val_dataloader):
                
                vinputs, vlabels = vdata['pixel_values'], vdata['labels']
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                
                voutputs = model(vinputs)
                
                # the class with the highest energy is what we choose as prediction
                predictions = voutputs.argmax(-1)
                correct_count += (predictions == vlabels).sum().item()
            
                vloss = criterian(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            print('Validation accuracy: {}'.format(correct_count / len(val_dataloader.dataset)))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss            
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)
                ## write model_path to a file
                with open("model_path.txt", "w+") as f:
                    f.write(model_path)
                    
            epoch_number += 1
 


        
## Inferencing loop
def inference():
    
    ## read model_path from a file
    with open("model_path.txt", "r") as f:
        model_path = f.read().strip()
     
    if model_path == "":
        print("No model path found")
        return    
        
    ## load the best model
    model.load_state_dict(torch.load(model_path))
    
    model.eval()
    with torch.no_grad():
        ## start time
        start_time = datetime.now()
        
        running_test_loss=0.
        correct_count = 0 
        total_correct_count = 0
        total_labels = 0
        for i, data in enumerate(test_dataloader):
            test_data, test_labels = data['pixel_values'], data['labels']
            test_data, test_labels = test_data.to(device), test_labels.to(device)
            
            test_outputs = model(test_data)
            total_labels += len(test_labels)
            
            # the class with the highest energy is what we choose as prediction
            predictions = test_outputs.argmax(-1)
            correct_count = (predictions == test_labels).sum().item()
            
            total_correct_count += correct_count
            total_labels += test_data.shape[0]
            
            if i % 100 == 99:
                ## every 100 steps, print the loss and accuracy
                print('Batch {} test loss: {} test accuracy {}'.format(i + 1, running_test_loss / (i + 1), correct_count / test_data.shape[0]))
                writer.add_scalar('Accuracy/test', correct_count / test_data.shape[0], i)
                
            test_loss = criterian(test_outputs, test_labels)
            running_test_loss += test_loss
            
        
        writer.flush()
        writer.close()
            
        endtime = datetime.now()
        print('Test accuracy: {}'.format(total_correct_count / total_labels))
        print('Total Inference time: {}'.format(endtime - start_time))   ## inference time in seconds  
        
import sys          
if __name__ == "__main__":
    task = sys.argv[1]
    if task == 'train':
        train_step()
    elif task == 'inference':
        inference()    
    
    
    

            
        
               
        
