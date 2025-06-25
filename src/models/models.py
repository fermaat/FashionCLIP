import torch
import time

import torch.nn as nn

from tqdm import tqdm
from transformers import CLIPVisionConfig, CLIPVisionModel

class ImageEncoderNetwork(nn.Module):
    """
    This is a generic class for using a clip feature extractor as a layer 
    in order to extract the normalized embeddings
    """
    def __init__(self):
        super(ImageEncoderNetwork, self).__init__()
        configuration = CLIPVisionConfig()

        # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
        self.vision_model = CLIPVisionModel(configuration)
        self.visual_projection = nn.Linear(in_features=768, out_features=512, bias=False)


    def load_from_clip(self, original_clip):
        self.vision_model = original_clip.vision_model
        self.visual_projection = original_clip.visual_projection

    def forward(self, x):
        vision_outputs = self.vision_model(x)['pooler_output']
        image_embeds = self.visual_projection(vision_outputs) 
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds



def get_lr(optimizer):
    """
    extracts the lr from the given optimizer (just to be displayed)
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Will train an epoch of the model
    """
    model.train()
    nb_batches = len(train_loader)
    tqdm_object = tqdm(train_loader, total=len(train_loader))   
    epoch_loss = 0.0
    for i, batch in enumerate(tqdm_object):
        anchor_img = batch['anchor_image']['pixel_values'].to(device)
        positive_img = batch['pos_image']['pixel_values'].to(device)
        negative_img = batch['neg_image']['pixel_values'].to(device)
        semipos = batch['semipos'].to(device)

        optimizer.zero_grad()# with torch.no_grad():
        anchor_out = model(anchor_img.squeeze())
        positive_out = model(positive_img.squeeze())
        negative_out = model(negative_img.squeeze())
        
        loss = criterion(anchor_out, positive_out, negative_out, semipos)
        # print(f'Iteration loss: {loss}')
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

        tqdm_object.set_postfix(
          batch="{}/{}".format(i+1,nb_batches),
          train_loss=loss.item(),
          lr=get_lr(optimizer))
    epoch_loss = epoch_loss / nb_batches
    return epoch_loss


def valid_epoch(model, dev_loader, criterion, device):
    """
    Will perform a validation loop on the given dataloader
    """
    model.eval()
    nb_batches = len(dev_loader)
    tqdm_object = tqdm(dev_loader, total=len(dev_loader))
    epoch_loss = 0.0   
    for i, batch in enumerate(tqdm_object):
        anchor_img = batch['anchor_image']['pixel_values'].to(device)
        positive_img = batch['pos_image']['pixel_values'].to(device)
        negative_img = batch['neg_image']['pixel_values'].to(device)
        semipos = batch['semipos'].to(device)

        anchor_out = model(anchor_img.squeeze())
        positive_out = model(positive_img.squeeze())
        negative_out = model(negative_img.squeeze())
        
        loss = criterion(anchor_out, positive_out, negative_out, semipos)
        epoch_loss += loss.item()
        tqdm_object.set_postfix(
          batch="{}/{}".format(i+1,nb_batches),
          dev_loss=loss.item(),
          )
    epoch_loss = epoch_loss / nb_batches
    return epoch_loss


def learning_loop(model, device, 
                  optimizer, lr_scheduler, criterion, 
                  max_epochs, max_bad_epochs, 
                  train_dataloader, val_dataloader):
    model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=CFG.patience, factor=CFG.factor)

    best_dev_score = float('inf')
    train_history = []
    dev_history = []
    nb_bad_epochs = 0

    print("Learning phase")
    print('Used device:', device)
    print("--------------")
    for epoch in range(1, max_epochs+1):

        print("Epoch {:03d}/{:03d}".format(epoch, max_epochs))

        if nb_bad_epochs >= max_bad_epochs:
            print("Epoch {:03d}/{:03d}: exiting training after too many bad epochs.".format(epoch, max_epochs))
            torch.save(model.state_dict(), "final.pt")
            break

        else:

            epoch_start_time = time.time()
            epoch_train_loss = train_epoch(model=model, train_loader=train_dataloader, optimizer=optimizer, criterion=criterion, device=device)
            with torch.no_grad():
                epoch_dev_score = valid_epoch(model=model, dev_loader=val_dataloader, criterion=criterion, device=device)

            duration = time.time() - epoch_start_time

            if lr_scheduler is not None:
                lr_scheduler.step(epoch_dev_score)

            train_history.append(epoch_train_loss)
            dev_history.append(epoch_dev_score)

            if epoch_dev_score < best_dev_score:
                nb_bad_epochs = 0
                best_dev_score = epoch_dev_score
                torch.save(model.state_dict(), "best.pt")
                print("Finished epoch {:03d}/{:03d} - Train loss: {:.7f} - Valid loss: {:.7f} - SAVED (NEW) BEST MODEL. Duration: {:.3f} s".format(
                epoch, max_epochs, epoch_train_loss, epoch_dev_score, duration))
            else:
                nb_bad_epochs += 1
                print("Finished epoch {:03d}/{:03d} - Train loss: {:.7f} - Valid loss: {:.7f} - NUMBER OF BAD EPOCH.S: {}. Duration: {:.3f} s".format(
                epoch, max_epochs, epoch_train_loss, epoch_dev_score, nb_bad_epochs, duration))
    
    history = {'train':train_history,'dev':dev_history}
    return history
