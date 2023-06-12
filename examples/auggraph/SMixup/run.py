import argparse
from dig.auggraph.method.SMixup.smixup import smixup

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IMDBB',
                    choices=['IMDBB','PROTEINS', 'MUTAG', "REDDITB", 'IMDBM', 'REDDITM5', 'REDDITM12', 'NCI1'])
parser.add_argument('--GMNET_nlayers', type=int, default=5,
                    help='Number of layers of GMNET')
parser.add_argument('--GMNET_hidden', type=int, default=100,
                    help='Number of hidden units of GMNET')
parser.add_argument('--GMNET_bs', type=int, default=32,
                    help='Batch size of training GMNET')
parser.add_argument('--GMNET_lr', type=float, default=1e-3,
                    help='Initial learning rate of GMNET')
parser.add_argument('--GMNET_epochs', type=int, default=10,
                    help='Number of epochs to train GMNET')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size during training')
parser.add_argument('--model', type = str, default='GIN', 
                    choices=['GCN', 'GIN'])   
parser.add_argument('--nlayers', type = int, default = 4,
                    help='Number of GNN layers.')  
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type = float, default = 0.2,
                    help='Dropout ratio.') 
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train the classifier.')
parser.add_argument('--alpha', type = float, default = 1.0,
                    help='mixup ratio.') 
parser.add_argument('--ckpt_path', type=str, default='../../../../test/ckpts/', 
                    help='Location for saving checkpoints')

args = parser.parse_args()

GMNET_conf = {}
GMNET_conf['nlayers'] = args.GMNET_nlayers
GMNET_conf['nhidden'] = args.GMNET_hidden
GMNET_conf['bs'] = args.GMNET_bs
GMNET_conf['lr'] = args.GMNET_lr
GMNET_conf['epochs'] = args.GMNET_epochs
      
runner = smixup('../../../../test/datasets', args.dataset, GMNET_conf)

runner.train_test(args.batch_size, args.model, cls_nlayers=args.nlayers, 
                  cls_hidden=args.hidden, cls_dropout=args.dropout, cls_lr=args.lr,
                  cls_epochs=args.epochs, alpha=args.alpha, ckpt_path=args.ckpt_path,)
