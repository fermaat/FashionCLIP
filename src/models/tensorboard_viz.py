import torch
import numpy as np
import os
import cv2
import pickle
from datasets import Dataset

import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter


def create_folder(folder, folder_purpose):
    """
    Just creates a folder if it doesn't exist
    """
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print(f"Warning: {folder_purpose} already exists")
        print("logging directory: " + str(folder))


class DeepFeatures(torch.nn.Module):    
    """
    This class extracts, reads, and writes data embeddings using a pretrained deep neural network. Meant to work with 
    Tensorboard's Embedding Viewer (https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin).
    When using with a 3 channel image input and a pretrained model from torchvision.models please use the 
    following pre-processing pipeline:
    
    transforms.Compose([transforms.Resize(imsize), 
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) ## As per torchvision docs
    
    Args:
        model (nn.Module): A Pytorch model that returns an (B,1) embedding for a length B batched input
        data_folder (str): The folder path where the input data elements should be written to
        tensorboard_folder (str): The folder path where the resulting Tensorboard log should be written to
        experiment_name (str): The name of the experiment to use as the log name
    """
    def __init__(self, model,
                 data_folder, 
                 tensorboard_folder,
                 experiment_name=None,
                 device='cpu'):
        
        super(DeepFeatures, self).__init__()
        
        self.model = model
        self.model.eval()

        self.experiment_name = experiment_name

        self.data_folder = data_folder
        self.imgs_folder = f'{self.data_folder}/imgs_{self.experiment_name}'
        self.embs_folder = f'{self.data_folder}/embs_{self.experiment_name}'
        self.tensorboard_folder = tensorboard_folder
        
        self.device = device
        
        self.writer = None
        
        
    def save_data(self, embeddings, images):
        """
        Will save embeddings and images data
        """
        create_folder(self.data_folder, 'data folder')
        create_folder(self.imgs_folder, 'images folder')
        create_folder(self.embs_folder, 'embedding folder')
        file_paths = []
        for i, image in enumerate(images):
            file_path = f'{self.imgs_folder}/{i}.png'
            image.save(file_path)
            file_paths.apend(file_path)

        result = Dataset.from_dict({'embeddings': embeddings, 'file_paths': file_paths})
        
        with open('data/train_embeds.pkl', 'wb') as f:
            pickle.dump(result, f)

    
    def generate_embeddings(self, x):
        """
        Generate embeddings for an input batched tensor
        
        Args:
            x (torch.Tensor) : A batched pytorch tensor
            
        Returns:
            (torch.Tensor): The output of self.model against x
        """
        anchor_img = x['anchor_image']['pixel_values'].to(self.device)
        positive_img = x['pos_image']['pixel_values'].to(self.device)
        negative_img = x['neg_image']['pixel_values'].to(self.device)
        # semipos = x['semipos'].to(self.device)
        with torch.no_grad():
            anchor_out = self.model(anchor_img.squeeze())
            positive_out = self.model(positive_img.squeeze())
            negative_out = self.model(negative_img.squeeze())

        return {**x,
                **{'anchor_out': anchor_out,
                   'positive_out': positive_out,
                   'negative_out': negative_out}}
    

    def generate_and_log_embeddings(self, input_batch):
        """
        Will generate embeddings from teh input batch and add them to the 
        tb log
        """
        # T.ToPILImage(tensor).resize(outsize.save(file_path))
        embs = self.generate_embeddings(input_batch)

        if self.writer is None:
            self._create_writer(self.experiment_name)

        self.writer.add_embedding(embs['anchor_out'], # , # torch.Size([64, 512])
                                  metadata=embs['caption'],
                                  label_img=embs['anchor_image']['pixel_values'].squeeze())
        self.writer.close()
        

    def _create_writer(self, name):
        """
        Create a TensorboardX writer object given an experiment name and assigns it to self.writer
        
        Args:
            name (str): Optional, an experiment name for the writer, defaults to self.experiment_name
        
        Returns:
            (bool): True if writer was created succesfully
        
        """
        if self.experiment_name is None:
            name = 'Experiment_' + str(np.random.random())
        else:
            name = self.experiment_name
        
        log_dir = os.path.join(self.tensorboard_folder, 
                               name)
        create_folder(log_dir, 'logfile')
        self.writer = SummaryWriter(log_dir=log_dir)
        
        return True
