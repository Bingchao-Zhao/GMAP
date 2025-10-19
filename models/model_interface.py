
import inspect
import importlib
import random
import pandas as pd

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from my_utils.utils import just_dir_of_file
from utils.utils import cross_entropy_torch

#---->
import torch
import torchmetrics
from my_utils.utils import *

#---->
import pytorch_lightning as pl
import my_utils.file_util as fu

# import monai
class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        self.kargs = kargs
        self.get_cam = True
        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.load_model()
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes),
                                                     torchmetrics.F1(num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes)])
        else : 
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'macro', task="multiclass")
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,
                                                                           average = 'micro', task="multiclass"),
                                                     torchmetrics.CohenKappa(num_classes = 2, task="multiclass"),
                                                     torchmetrics.F1Score(num_classes = 2,
                                                                     average = 'macro', task="multiclass"),
                                                     torchmetrics.Recall(average = 'macro',#multiclass=False,
                                                                         num_classes = 2, task="multiclass"),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = 2, task="multiclass"),
                                                    torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = 2, task="multiclass")])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0
        self.val_step_outputs = []
        self.output_results = []
        self.error_wsi = []


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->inference
        data, label = batch[0:2]
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']

        #---->loss
        loss = self.loss(logits, label)

        Y_hat = int(Y_hat)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss} 

    def on_train_epoch_end(self):
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print(' Train class {}: acc {}, correct {}/{}\n'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def validation_step(self, batch, batch_idx):
        data, label = batch[0:2]
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)
        self.val_step_outputs.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label})
        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    def on_validation_epoch_end(self):
        logits = torch.cat([x['logits'] for x in self.val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in self.val_step_outputs], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in self.val_step_outputs])
        target = torch.stack([x['label'] for x in self.val_step_outputs], dim = 0)
        
        #---->
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        torch.use_deterministic_algorithms(False)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)
        torch.use_deterministic_algorithms(True)
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print(' Train class {}: acc {}, correct {}/{}\n'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        data, label, index, npy_name, patient = batch

        data = torch.autograd.Variable(data, requires_grad=True)
        self.model = self.model.eval()
        results_dict = self.model(data=data, label=label)
        
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        index = index[0].detach().cpu().numpy()

        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)
        if Y_hat.item() != Y:
            self.error_wsi.append([get_name_from_path(npy_name[0]), Y_hat.item(),Y])
        self.output_results.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'patient':patient})
        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'patient':patient}

    def on_test_epoch_end(self):
        probs = torch.cat([x['Y_prob'] for x in self.output_results], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in self.output_results])
        target = torch.stack([x['label'] for x in self.output_results])
        patient = [x['patient'][0] for x in self.output_results]
        #---->
        auc = self.AUROC(probs, target.squeeze())
        csv_context = []
        for i in range(len(target.squeeze())):
            csv_context.append([patient[i], probs[i][1].cpu().detach().numpy(), target.squeeze()[i].cpu().detach().numpy()])
        torch.use_deterministic_algorithms(False)
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc


        save_csv_path = '{}/{}/{}/{}/{}/MGAP/{}_{}_best_auc.csv'.format(
            self.kargs['cfg']['conf_log_path'],
            self.kargs['cfg']['mod_name'],
            self.kargs['cfg']['extractor'],
            self.kargs['cfg']['infer_ds'],
            self.kargs['cfg']['gen_type'],
            self.kargs['cfg']['infer_ds'],
            self.kargs['cfg']['gen_type'])
        just_dir_of_file(save_csv_path)
        fu.write_csv_row(save_csv_path, csv_context, model='w')

        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        print()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->
        result = pd.DataFrame([metrics])
        save_csv_path = '{}/{}/{}/{}/{}/MGAP/{}_{}_result.csv'.format(
            self.kargs['cfg']['conf_log_path'],
            self.kargs['cfg']['mod_name'],
            self.kargs['cfg']['extractor'],
            self.kargs['cfg']['infer_ds'],
            self.kargs['cfg']['gen_type'],
            self.kargs['cfg']['infer_ds'],
            self.kargs['cfg']['gen_type'])
        result.to_csv(save_csv_path)
        


    def load_model(self):
        if self.kargs.get('cfg') is None:
            name = self.kargs['mod_name']
        else:
            name = self.kargs['cfg']['mod_name']#self.hparams.model.name

        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):

        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)