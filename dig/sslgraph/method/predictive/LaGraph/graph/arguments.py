import argparse

def arg_parse_graph():
    parser = argparse.ArgumentParser(description='LaGraph Graph Tasks.')
    parser.add_argument('--data_dir', default='./data/', type=str, help='Directory to read data')
    parser.add_argument('--save_dir', default='results/exp1/', type=str, help='Directory to save models & records')
    parser.add_argument('--save_model', default=False, type=bool, help='Set True if save pretrained models')
    parser.add_argument('--DS', dest='DS', default='MUTAG', help='Dataset')
    parser.add_argument('--th', type=int, default=-1, help='Threshold of degree. Set -1 if not needed')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', default=1e-5, dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--alpha', default=10, type=float, help='Weight of loss_inv')
    parser.add_argument('--mratio', default=0.1, type=float, help='Mask ratio')
    parser.add_argument('--mstd', default=1, type=float, help='Mask std dev')
    parser.add_argument('--mmode', default='whole', type=str, help='Mask whole or partial node')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='')  
    parser.add_argument('--enly', type=int, default=3, help='Number of encoder layer')
    parser.add_argument('--dely', type=int, default=2, help='Number of decoder layer')
    parser.add_argument('--decoder', default='mlp', type=str, help='Type of decoder')
    parser.add_argument('--interval', type=int, default=1, help='Interval to check performance')
    parser.add_argument('--pool', default='add', type=str, help='Global pooling')
    parser.add_argument('--loss', default='mse', type=str, help='Type of pretrain loss')
    parser.add_argument('--bsz', type=int, default=128, help='Batch size')

    return parser.parse_args()

