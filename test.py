import util
import argparse
import torch
import torch.nn.functional as F
from model import GWNet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(args, save_pred_path='preds.csv', save_metrics_path='last_test_metrics.csv', loader='test', **model_kwargs):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    data = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, n_obs=args.n_obs, fill_zeroes=args.fill_zeroes)
    scaler = data['scaler']
    realy = torch.Tensor(data[f'y_{loader}']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    print(realy.shape)

    adjinit, supports = util.make_graph_inputs(args, device)
    model = GWNet.from_args(args, device, supports, adjinit, **model_kwargs)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    print('model loaded successfully')

    met_df, yhat = util.calc_tstep_metrics(model, device, data[f'{loader}_loader'], scaler, realy, args.seq_length)
    df2 = util.make_pred_df(realy, yhat, scaler, args.seq_length)
    met_df.to_csv(save_metrics_path)
    df2.to_csv(save_pred_path, index=False)
    if args.plotheatmap: 
        plot_learned_adj_matrix(model)
    return met_df, df2

def plot_learned_adj_matrix(model):
    adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
    adp = adp.cpu().detach().numpy()
    adp = adp / np.max(adp)
    df = pd.DataFrame(adp)
    sns.heatmap(df, cmap="RdYlBu")
    plt.savefig("heatmap.png")

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
    parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
    parser.add_argument('--do_graph_conv', action='store_true',
                        help='whether to add graph convolution layer')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true',
                        help='whether random initialize adaptive adj')
    parser.add_argument('--seq_length', type=int, default=12, help='')
    parser.add_argument('--nhid', type=int, default=40, help='Number of channels for internal conv')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--n_obs', default=None, help='Only use this many observations. For unit testing.')
    parser.add_argument('--apt_size', default=10, type=int)
    parser.add_argument('--cat_feat_gc', action='store_true')
    parser.add_argument('--fill_zeroes', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='')
    parser.add_argument('--plotheatmap', action='store_true')
    args = parser.parse_args()
    main(args)
