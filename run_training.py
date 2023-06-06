# for command line args and json config file parsing
import sys
import json
# for linking .py files
from models.neural_net import *
from models.classification_transformer import *
from lightning_modules import *
from torch.utils.data import DataLoader

# for logging results
import wandb
from utils.padding_collate_fn import *
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


##################################
model_type="nn" #nn
prediction_type="prob" #prob
##################################

configfile = f"configs/{model_type}_{prediction_type}_config.json" 
with open(configfile) as stream:
    config = json.load(stream)

model_config = config['model_config']
loss_config = config['loss_config']
optim_config = config['optim_config']
trainer_config = config['trainer_config']


data_type= "seq" if model_type=="transformer" else "kmer"

seq_train_set = torch.load(f"data/{data_type}_train_{prediction_type}_r.pt")
seq_val_set = torch.load(f"data/{data_type}_val_{prediction_type}_r.pt")
seq_test_set = torch.load(f"data/{data_type}_test_{prediction_type}_r.pt")


# load data 
train_dl = DataLoader(seq_train_set, collate_fn=padding_collate, batch_size=config['batch_size'], shuffle=True)
val_dl = DataLoader(seq_val_set, collate_fn=padding_collate, batch_size=config['batch_size'], shuffle=True) 
test_dl = DataLoader(seq_test_set, collate_fn=padding_collate, batch_size=config['batch_size'], shuffle=True) 

# instance model
model = eval(model_config['model_name'])(model_config['model_kwargs'])

# instance litmodelwrapper
litmodel = LitModelWrapper(model=model, loss_config=loss_config, optim_config=optim_config)

# instance wandb logger
plg= WandbLogger(project = config['wandb_project'],
                 entity = 'emmazchen', 
                 config=config) ## include run config so it gets logged to wandb 
plg.watch(litmodel) ## this logs the gradients for model 

## add the logger object to the training config portion of the run config 
trainer_config['logger'] = plg

## set to save every checkpoint (lightning saves the best checkpoint of model by default)
checkpoint_cb = ModelCheckpoint(save_top_k=-1, every_n_epochs = None, every_n_train_steps = None, train_time_interval = None)
trainer_config['callbacks'] = [checkpoint_cb]
trainer = pl.Trainer(**trainer_config)

# dry run lets you check if everythign can be loaded properly 
if config['dryrun']:
    print("Successfully loaded everything. Quitting")
    sys.exit()

# train
trainer.fit(litmodel, train_dataloaders = train_dl, val_dataloaders=val_dl)

# predict
out = trainer.predict(litmodel, dataloaders = test_dl)
