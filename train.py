import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

import pytorch_lightning as pl
from pytorch_lightning import Trainer

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='test', type=str,
                        help='train, test')
    parser.add_argument('--gen_type', default='IDH', type=str,
                        help='TERT, IDH, 1p19q, 7g10l')
    parser.add_argument('--mod_name', default='GMAP', type=str)
    parser.add_argument('--extractor', default='UNI' , type=str,
                        help='ResNet50 UNI')
    parser.add_argument('--ds', default='TCGA', type=str,help="Training dataset.") 
    parser.add_argument('--gpus', default = [0])
    parser.add_argument('--infer_ds', default='TCGA', type=StopIteration)
    parser.add_argument('--config', default='config/GMAP.yaml',type=str)
    
    args = parser.parse_args() 
    return args 


def main(cfg):

    pl.seed_everything(cfg.General.seed)

    cfg.load_loggers = load_loggers(cfg)

    cfg.callbacks = load_callbacks(cfg)

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


    ModelInterface_dict = {'model': cfg.Model,
                            'General':cfg.General,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            "extractor" :cfg.extractor,
                            "mod_name" :cfg.mod_name
                            }

    trainer = Trainer(
                        num_sanity_val_steps=0, 
  
                        logger=cfg.load_loggers,
                        callbacks=cfg.callbacks,
                        max_epochs= cfg.General.epochs,  
                        precision=cfg.General.precision,  
                        accumulate_grad_batches=cfg.General.grad_acc,
                        deterministic=True,
                        check_val_every_n_epoch=1,
                        detect_anomaly=True
                    )

    if cfg.General.server == 'train':
        
        model = ModelInterface(**ModelInterface_dict)
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = ModelInterface.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            a = trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':
    
    args = make_parse()
    cfg = read_yaml(args.config)

    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.gen_type = args.gen_type
    cfg.ds = args.ds
    cfg.infer_ds = args.infer_ds
    cfg.extractor = args.extractor
    cfg.mod_name = args.mod_name

    main(cfg)
 