"""GraphZoo trainer"""
from __future__ import division
from __future__ import print_function
import datetime
import json
import logging
from operator import ne
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from GraphZoo.graphzoo.optimizers.radam import RiemannianAdam
from GraphZoo.graphzoo.optimizers.rsgd import RiemannianSGD
from GraphZoo.graphzoo.config import parser
from GraphZoo.graphzoo.models.base_models import NCModel, LPModel
from GraphZoo.graphzoo.utils.train_utils import get_dir_name, format_metrics
from GraphZoo.graphzoo.dataloader.dataloader import DataLoader
from GraphZoo.graphzoo.dataloader.download import download_and_extract
from GraphZoo.graphzoo.manifolds import PoincareBall, Hyperboloid

from min_norm_solvers import gradient_normalizers, MinNormSolver

class Trainer:
    """
    GraphZoo Trainer

    Input Parameters
    ----------
        'lr': (0.05, 'initial learning rate (type: float)'),
        'dropout': (0.0, 'dropout probability (type: float)'),
        'cuda': (-1, 'which cuda device to use or -1 for cpu training (type: int)'),
        'repeat': (10, 'number of times to repeat the experiment (type: int)'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam, RiemannianSGD] (type: str)'),
        'epochs': (5000, 'maximum number of epochs to train for (type:int)'),
        'weight-decay': (0.0, 'l2 regularization strength (type: float)'),
        'momentum': (0.999, 'momentum in optimizer (type: float)'),
        'patience': (100, 'patience for early stopping (type: int)'),
        'seed': (1234, 'seed for training (type: int)'),
        'log-freq': (5, 'how often to compute print train/val metrics in epochs (type: int)'),
        'eval-freq': (1, 'how often to compute val metrics in epochs (type: int)'),
        'save': (0, '1 to save model and logs and 0 otherwise (type: int)'),
        'save-dir': (None, 'path to save training logs and model weights (type: str)'),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant (type: int)'),
        'gamma': (0.5, 'gamma for lr scheduler (type: float)'),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping (type: float)'),
        'min-epochs': (100, 'do not early stop before min-epochs (type: int)'),
        'betas': ((0.9, 0.999), 'coefficients used for computing running averages of gradient and its square (type: Tuple[float, float])'),
        'eps': (1e-8, 'term added to the denominator to improve numerical stability (type: float)'),
        'amsgrad': (False, 'whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond` (type: bool)'),
        'stabilize': (None, 'stabilize parameters if they are off-manifold due to numerical reasons every ``stabilize`` steps (type: int)'),
        'dampening': (0,'dampening for momentum (type: float)'),
        'nesterov': (False,'enables Nesterov momentum (type: bool)')
        
    API Input Parameters
    ----------
        args: list of above defined input parameters from `graphzoo.config`
        optimizer: a :class:`optim.Optimizer` instance
        model: a :class:`BaseModel` instance
    
    """
    def __init__(self,args,model, optimizer,data):

        self.args=args
        self.model=model
        self.optimizer =optimizer
        self.data=data
        self.best_test_metrics = None
        self.best_emb = None
        self.best_val_metrics = self.model.init_metric_dict()

        self.pretrain_epoch = 2000

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if int(self.args.cuda) >= 0:
            torch.cuda.manual_seed(self.args.seed)
            
        if args.cuda is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'    
    
        logging.getLogger().setLevel(logging.INFO)
        if self.args.save:
            if not self.args.save_dir:
                dt = datetime.datetime.now()
                date = f"{dt.year}_{dt.month}_{dt.day}"
                models_dir = os.path.join(os.getcwd(), self.args.dataset, self.args.task,self.args.model, date)
                self.save_dir = get_dir_name(models_dir)
            else:
                self.save_dir = self.args.save_dir
            logging.basicConfig(level=logging.INFO,
                                handlers=[
                                    logging.FileHandler(os.path.join(self.save_dir, 'log.txt')),
                                    logging.StreamHandler()
                                ])

        logging.info(f'Using: {device}')
        logging.info("Using seed {}.".format(self.args.seed))


        if not self.args.lr_reduce_freq:
            self.lr_reduce_freq = self.args.epochs

        # self.lr_reduce_freq = 200

        logging.info(str(self.model))
    
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(self.lr_reduce_freq),
            gamma=float(self.args.gamma)
        )
        tot_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        if self.args.cuda is not None and int(self.args.cuda) >= 0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.cuda)
            self.model = self.model.to(device)
            for x, val in self.data.items():
                if torch.is_tensor(self.data[x]):
                    self.data[x] = self.data[x].to(device)
  
    def run(self):
        """
        Train model.

        The processes:
            Run each epoch -> Run scheduler -> Should stop early?
        """
        t_total = time.time()
        counter = 0

        # load model
        model_dir = 'models'
        if self.args.exist_epoch !=0:
            init_filename = 'HGCNAE_' + str(self.args.exist_epoch) + '.ckpt'
            init_model_url = os.path.join(model_dir, init_filename)
            checkpoint = torch.load(init_model_url)
            self.model.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        preserve_embeddings = None
        preserve_recon_x = None
        
        if self.args.exist_epoch == self.args.epochs:
            self.model.eval()
            embeddings = self.model.encode(self.data['features'], self.data['adj_train_norm'])
            if self.args.recon:
                recon_x, val_metrics = self.model.compute_metrics(embeddings, self.data, 'val')
            else:
                val_metrics = self.model.compute_metrics(embeddings, self.data, 'val')


        for epoch in range(self.args.exist_epoch, self.args.epochs):
            # if (epoch+1)%100==0:
            #     filename = 'HGCNAE_' + str(epoch + 1) + '.ckpt'
            #     model_url = os.path.join(model_dir, filename)
            #     state = {'net':self.model.state_dict(),'optimizer':self.optimizer.state_dict()}
            #     torch.save(state, model_url)

            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            # embeddings = self.model.encode(self.data['features'], self.data['adj_train_norm'])
            # if self.args.recon:
            #     recon_x, train_metrics = self.model.compute_metrics(embeddings, self.data, 'train')
            # else:
            #     train_metrics = self.model.compute_metrics(embeddings, self.data, 'train')
                
            ### caculate train loss ###
            loss_data = {}
            grads = {}

            # recon loss combine
            loss = self.model.recon_loss_combine(self.data['features'], self.data['adj_train_norm'])
            grads['recon_loss_combine'] = []
            loss_data['recon_loss_combine'] = loss.data
            loss.backward()
            for param in self.model.big_model.parameters():
                if param.grad is not None:
                    grads['recon_loss_combine'].append(param.grad.data.detach().cpu())
            self.model.zero_grad()

            # recon loss eu
            loss = self.model.recon_loss_eu(self.data['features'], self.data['adj_train_norm'])
            grads['recon_loss_eu'] = []
            loss_data['recon_loss_eu'] = loss.data
            loss.backward()
            for param in self.model.big_model.parameters():
                if param.grad is not None:
                    grads['recon_loss_eu'].append(param.grad.data.detach().cpu())
            self.model.zero_grad()

            # recon loss po
            loss = self.model.recon_loss_eu(self.data['features'], self.data['adj_train_norm'])
            grads['recon_loss_po'] = []
            loss_data['recon_loss_po'] = loss.data
            loss.backward()
            for param in self.model.big_model.parameters():
                if param.grad is not None:
                    grads['recon_loss_po'].append(param.grad.data.detach().cpu())
            self.model.zero_grad()

            # recon loss lo
            loss = self.model.recon_loss_lo(self.data['features'], self.data['adj_train_norm'])
            grads['recon_loss_lo'] = []
            loss_data['recon_loss_lo'] = loss.data
            loss.backward()
            for param in self.model.big_model.parameters():
                if param.grad is not None:
                    grads['recon_loss_lo'].append(param.grad.data.detach().cpu())
            self.model.zero_grad()

            # caculate weights
            tasks = ['recon_loss_eu', 'recon_loss_po', 'recon_loss_lo', 'recon_loss_combine']
            # tasks = ['recon_loss', 'cl_loss_1', 'cl_loss_2']
            gn = gradient_normalizers(grads, loss_data, 'l2')
            for m in loss_data:
                for gr_i in range(len(grads[m])):
                    grads[m][gr_i] = grads[m][gr_i] / gn[m].to(grads[m][gr_i].device)
            sol, _ = MinNormSolver.find_min_norm_element_FW([grads[m] for m in tasks])
            sol = {k:sol[i] for i, k in enumerate(tasks)}
            # sol['recon_loss_po'] = 1
            # self.model.zero_grad()

            train_loss = 0
            embeddings = self.model.encode(self.data['features'], self.data['adj_train_norm'])
            loss_dict = self.model.res

            for i, l in loss_dict.items():
                train_loss += float(sol[i]) * l

            # end adding
            train_loss.backward()


            # caculate train metrics for presentation
            train_metrics = {}
            train_metrics['loss'] = train_loss
            for task in tasks:
                train_metrics[task] = loss_data[task]
                train_metrics[task+'_weight'] = sol[task]

            embeddings = self.model.h

            # train_metrics['loss'].backward()
            if self.args.grad_clip is not None:
                max_norm = float(self.args.grad_clip)
                all_params = list(self.model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            if (epoch + 1) % self.args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                    'lr: {}'.format(self.lr_scheduler.get_lr()[0]),
                                    format_metrics(train_metrics, 'train'),
                                    'time: {:.4f}s'.format(time.time() - t)
                                    ]))
            if (epoch + 1) % self.args.eval_freq == 0:
                self.model.eval()
                embeddings = self.model.encode(self.data['features'], self.data['adj_train_norm'])
                if self.args.recon:
                    recon_x, val_metrics = self.model.compute_metrics(embeddings, self.data, 'val')
                else:
                    val_metrics = self.model.compute_metrics(embeddings, self.data, 'val')
                if (epoch + 1) % self.args.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
                if self.model.has_improved(self.best_val_metrics, val_metrics):
                    # self.best_test_metrics = self.model.compute_metrics(embeddings, self.data, 'test')
                    # self.best_emb = embeddings.cpu()
                    # if self.args.save:
                    #     np.save(os.path.join(self.save_dir, 'embeddings.npy'), self.best_emb.detach().numpy())
                    # self.best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == self.args.patience and epoch > self.args.min_epochs:
                        logging.info("Early stopping")
                        break
                    

        logging.info("Optimization Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        if self.args.recon:
            return embeddings, recon_x
        else:
            return embeddings

    def evaluate(self):
        """
        Evaluate the model.
        """
        if not self.best_test_metrics:
            self.model.eval()
            self.best_emb = self.model.encode(self.data['features'], self.data['adj_train_norm'])
            self.best_test_metrics = self.model.compute_metrics(self.best_emb, self.data, 'test')
        logging.info(" ".join(["Val set results:", format_metrics(self.best_val_metrics, 'val')]))
        logging.info(" ".join(["Test set results:", format_metrics(self.best_test_metrics, 'test')]))
        if self.args.save:
            np.save(os.path.join(self.save_dir, 'embeddings.npy'), self.best_emb.cpu().detach().numpy())
            if hasattr(self.model.encoder, 'att_adj'):
                filename = os.path.join(self.save_dir, self.args.dataset + '_att_adj.p')
                pickle.dump(self.model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
                print('Dumped attention adj: ' + filename)
            
            json.dump(vars(self.args), open(os.path.join(self.save_dir, 'config.json'), 'w'))
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pth'))
            logging.info(f"Saved model in {self.save_dir}")
        return self.best_test_metrics


if __name__ == '__main__':

    """
    Main function to run command line evaluations

    Note
    ----------
    Metrics averaged over repetitions are F1 score for node classification (accuracy for cora and pubmed),
    ROC for link prediction. Metrics to be averaged can be changed easily in the code.
    """
    args = parser.parse_args()
    result_list=[]
    
    args = parser.parse_args()
        
    download_and_extract(args)

    data=DataLoader(args)

    for i in range(args.repeat):
        
        if args.task=='nc':
            model=NCModel(args)
        else:
            model=LPModel(args)

        if args.optimizer=='Adam':
            optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                                   betas=args.betas, eps=args.eps, amsgrad=args.amsgrad)
        if args.optimizer =='RiemannianAdam':
            optimizer=RiemannianAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    betas=args.betas, eps=args.eps ,amsgrad=args.amsgrad, 
                                    stabilize=args.stabilize)
        if args.optimizer =='RiemannianSGD':
            optimizer=RiemannianSGD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum, dampening=args.dampening, nesterov=args.nesterov,
                                    stabilize=args.stabilize)

        trainer=Trainer(args,model, optimizer,data)
        trainer.run()
        result=trainer.evaluate()

        if args.task=='nc' and args.dataset in ['cora','pubmed']:
            result_list.append(100*result['acc'])

        elif args.task=='nc' and args.dataset not in ['cora','pubmed']:
            result_list.append(100*result['f1'])

        else:
            result_list.append(100*result['roc'])
            
    result_list=torch.FloatTensor(result_list)
    print("Score",torch.mean(result_list),"Error",torch.std(result_list))
