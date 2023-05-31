import torch 
from torch import nn
import lightning.pytorch as pl
 
def binary_accuracy_logits(preds, label):
    # apply sigmoid
    preds = nn.functional.sigmoid(preds)
    # apply threshold of 0.5
    one_preds = preds > .5
    # get accuracy
    accuracy = (one_preds == label).sum() / len(preds)
    return accuracy


class LitModelWrapper(pl.LightningModule): 
    def __init__(self, model, loss_config, optim_config, model_type):
        super().__init__()
        self.model = model
        self.xtype = 'kmer count' if model_type=='NeuralNet' else 'seq' if model_type=='Transformer' else print('Specify in lighning_modules whether to use seq or kmer count for this model type')
        self.loss_fn = eval(loss_config['loss_fn'])()
        self.optim_config = optim_config
        self.bin_acc = binary_accuracy_logits

    def forward(self, batch):
        x = batch[self.xtype]
        logit = self.model(x)
        return logit

    def training_step(self, batch, batch_idx):
        x = batch[self.xtype]
        label = batch['label']
        batch_size = label.shape[0]
        logit = self.model(x).squeeze()
        loss = self.loss_fn(logit, label)
        self.log('train_loss', loss, batch_size=batch_size)
        accuracy = self.bin_acc(logit, label)
        self.log ('train_accuracy', accuracy, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xtype]
        label = batch['label']
        batch_size = label.shape[0]
        logit = self.model(x).squeeze()
        loss = self.loss_fn(logit, label)
        self.log('validation_loss', loss, sync_dist = True, batch_size=batch_size)
        accuracy = self.bin_acc(logit, label)
        self.log ('validation_accuracy', accuracy, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        x = batch[self.xtype]
        label = batch['label']
        batch_size = label.shape[0]
        logit = self.model(x).squeeze()
        loss = self.loss_fn(logit, label)
        self.log('test_loss', loss, sync_dist = True, batch_size=batch_size)
        accuracy = self.bin_acc(logit, label)
        self.log ('test_accuracy', accuracy, batch_size=batch_size)



 #optimizer
    def configure_optimizers(self):
        optim_fn = eval(self.optim_config['optim_fn'])
        optimizer = optim_fn(self.parameters(), **self.optim_config['optim_kwargs'])
        ### You can also use a learning rate scheduler here, but Ive commented it out for simplicity
        # sched_fn = eval(self.optim_config["scheduler"]) 
        # scheduler = sched_fn(optimizer,  **self.optim_config['scheduler_kwargs'])
        # scheduler_config = {
        #     "scheduler": scheduler,
        #     "interval": "step",
        #     "name":"learning_rate"
        # }
        optimizer_dict = {"optimizer" : optimizer#, 
            #"lr_scheduler" : scheduler_config
         }
        return optimizer   # it was originally optimizer_dict
