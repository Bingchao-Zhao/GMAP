import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import numpy as np
import glob
import attr
from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='test', type=str,
                        help='train, test')
    parser.add_argument('--gen_type', default='1p19q', type=str,
                        help='TERT, IDH, 1p19q, 7g10l')
    parser.add_argument('--mod_name', default='TransMIL', type=str,
                        help='TransMILori  TransMIL. TransMIL is the GMAP. TransMILori is the TransMIL.')
    parser.add_argument('--extractor', default='ResNet50' , type=str,
                        help='ResNet50 UNI')
    parser.add_argument('--ds', default='TCGA', type=str,help='dataset. only work in shandong data.'+\
                        "'SDPH','TCGA','QMH', 'smu_tcga_combine', 'SWH', 'EBRAINS', 'SCH") 
    parser.add_argument('--gpus', default = [0])
    parser.add_argument('--infer_ds', default='QMH', type=str,help='dataset. only work in shandong data.'+\
                        " SDPH, GDPH, TCGA, QMH,"+\
                        " PUSZH, SWH, EBRAINS, SCH,"+\
                        " YCH, SMU, TCGA_LUNG, CAMELYON, THH, ZY, SJBH, YHDH") 

    parser.add_argument('--mod', default='ori', type=str)
    parser.add_argument('--config', default='Camelyon/TransMIL.yaml',type=str)
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--resolu', default='20X_256', type=str)
    
    args = parser.parse_args() 
    return args 

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,
                'gen_type':cfg.gen_type,
                'ds':cfg.ds,
                'infer_ds':cfg.infer_ds,
                "extractor" :cfg.extractor,
                "mod_name" :cfg.mod_name
                }
    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'General':cfg.General,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            "extractor" :cfg.extractor,
                            "mod_name" :cfg.mod_name
                            }
    # model = ModelInterface(**ModelInterface_dict)
    
    #---->Instantiate Trainer
    
    trainer = Trainer(
                        num_sanity_val_steps=0, 
                        # amp_backend='apex',
                        logger=cfg.load_loggers,
                        callbacks=cfg.callbacks,
                        max_epochs= cfg.General.epochs,
                        # gpus=cfg.General.gpus,
                        # amp_level=cfg.General.amp_level,  
                        precision=cfg.General.precision,  
                        accumulate_grad_batches=cfg.General.grad_acc,
                        deterministic=True,
                        check_val_every_n_epoch=1,
                        # inference_mode=False,
                        detect_anomaly=True
                    )
    #---->train or test
    if cfg.General.server == 'train':
        
        model = ModelInterface(**ModelInterface_dict)
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = ModelInterface.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            # new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            a = trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':
    

    args = make_parse()
    cfg = read_yaml(args.config)

    #---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold
    cfg.gen_type = args.gen_type
    cfg.resolu = args.resolu
    cfg.mod = args.mod
    cfg.ds = args.ds
    cfg.infer_ds = args.infer_ds
    cfg.extractor = args.extractor
    cfg.mod_name = args.mod_name



    #---->main
    main(cfg)
 