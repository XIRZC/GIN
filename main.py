import argparse
import statistics as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data, separate_data_allfolds
from models.graphcnn import GraphCNN
import os
from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
	
    # test mode
    parser.add_argument('--test', action="store_true",
                        help='In test mode, num_epochs is 3')
    # dataset agnostic args, always fixed
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')

    # dataset specific args
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    # model specific args
    parser.add_argument('--model', type=str, default="SUM-MLP-0",
                        help='name of model (default: SUM-MLP-0)')
    ## model configs args
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    ## training configs args
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')

    parser.add_argument('--tensorboard', type=str, default='output',
                        help='Tensorboard logs output dir (default: output)')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    output_dir = args.tensorboard
            
    # dataset config dict by {'$args.dataset': dataset_dict}
    dataset_config_dict = {
        # 5 Social Network Datasets
        'COLLAB': {
            'hidden_dim': 64,               # 64 for social graphs
            'batch_size': 32,               # 32 or 128
            'final_dropout': 0.5,           # 0 ~ 0.5
            'epochs': 350,                  # default 350
            'graph_pooling_type': 'mean',   # mean for social datasets
            'degree_as_tag': True           # degree features
        },
        'IMDBBINARY': {
            'hidden_dim': 64,               # 64 for social graphs
            'batch_size': 32,               # 32 or 128
            'final_dropout': 0.5,           # 0 ~ 0.5
            'epochs': 350,                  # default 350
            'graph_pooling_type': 'mean',   # mean for social datasets
            'degree_as_tag': True           # degree features
        },
        'IMDBMULTI': {
            'hidden_dim': 64,               # 64 for social graphs
            'batch_size': 32,               # 32 or 128
            'final_dropout': 0.5,           # 0 ~ 0.5
            'epochs': 350,                  # default 350
            'graph_pooling_type': 'mean',   # mean for social datasets
            'degree_as_tag': True           # degree features
        },
        'REDDITBINARY': {
            'hidden_dim': 64,               # 64 for social graphs
            'batch_size': 32,               # 32 or 128
            'final_dropout': 0.5,           # 0 ~ 0.5
            'epochs': 350,                  # default 350
            'graph_pooling_type': 'mean',   # mean for social datasets
            'degree_as_tag': True           # degree features
        },
        'REDDITMULTI5K': {
            'hidden_dim': 64,               # 64 for social graphs
            'batch_size': 32,               # 32 or 128
            'final_dropout': 0.5,           # 0 ~ 0.5
            'epochs': 350,                  # default 350
            'graph_pooling_type': 'mean',   # mean for social datasets
            'degree_as_tag': True           # degree features
        },
        # 4 Bioinformatics Datasets
        'MUTAG': {
            'hidden_dim': 32,               # 16 or 32 for bioinformatics graphs
            'batch_size': 32,               # 32 or 128
            'final_dropout': 0.5,           # 0 ~ 0.5
            'epochs': 350,                  # default 350
            'graph_pooling_type': 'sum',    # sum for bioinformatics datasets
            'degree_as_tag': False          # input categorical features
        },
        'PTC': {
            'hidden_dim': 32,               # 16 or 32 for bioinformatics graphs
            'batch_size': 32,               # 32 or 128
            'final_dropout': 0.5,           # 0 ~ 0.5
            'epochs': 350,                  # default 350
            'graph_pooling_type': 'sum',    # sum for bioinformatics datasets
            'degree_as_tag': False          # input categorical features
        },
        'NCI1': {
            'hidden_dim': 32,               # 16 or 32 for bioinformatics graphs
            'batch_size': 32,               # 32 or 128
            'final_dropout': 0.5,           # 0 ~ 0.5
            'epochs': 350,                  # default 350
            'graph_pooling_type': 'sum',    # sum for bioinformatics datasets
            'degree_as_tag': False          # input categorical features
        },
        'PROTEINS': {
            'hidden_dim': 32,               # 16 or 32 for bioinformatics graphs
            'batch_size': 32,               # 32 or 128
            'final_dropout': 0.5,           # 0 ~ 0.5
            'epochs': 350,                  # default 350
            'graph_pooling_type': 'sum',    # sum for bioinformatics datasets
            'degree_as_tag': False          # input categorical features
        },
    }
    # model config dict by {'$args.model': model_dict}
    model_config_dict = {
        'SUM-MLP-0': {
            'neighbor_pooling_type': 'sum',
            'learn_eps': False,
            'num_layers': 5,
            'num_mlp_layers': 2,
        },
        'SUM-MLP-epsilon': {
            'neighbor_pooling_type': 'sum',
            'learn_eps': True,
            'num_layers': 5,
            'num_mlp_layers': 2,
        },
        'SUM-1-LAYER': {
            'neighbor_pooling_type': 'sum',
            'learn_eps': False,
            'num_layers': 5,
            'num_mlp_layers': 1,
        },
        'MEAN-MLP': {
            'neighbor_pooling_type': 'mean',
            'learn_eps': False,
            'num_layers': 5,
            'num_mlp_layers': 2,
        },
        'MEAN-1-LAYER': {
            'neighbor_pooling_type': 'mean',
            'learn_eps': False,
            'num_layers': 5,
            'num_mlp_layers': 1,
        },
        'MAX-MLP': {
            'neighbor_pooling_type': 'max',
            'learn_eps': False,
            'num_layers': 5,
            'num_mlp_layers': 2,
        },
        'MAX-1-LAYER': {
            'neighbor_pooling_type': 'max',
            'learn_eps': False,
            'num_layers': 5,
            'num_mlp_layers': 1,
        },
    }

    dataset_list = ['COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'MUTAG', 'NCI1', 'PROTEINS', 'PTC', 'REDDITBINARY', 'REDDITMULTI5K']
    all_model_list = ['SUM-MLP-0', 'SUM-MLP-epsilon', 'SUM-1-LAYER', 'MEAN-MLP', 'MEAN-1-LAYER', 'MAX-MLP', 'MAX-1-LAYER']
    selected_model_list = ['SUM-MLP-0', 'SUM-MLP-epsilon', 'MEAN-1-LAYER', 'MAX-1-LAYER']

    writer = SummaryWriter(os.path.join(output_dir, args.dataset))

    print('-'*100)
    print(f"=> Dataset {args.dataset} loading...")
    dataset_config = dataset_config_dict[args.dataset]
    for key, value in dataset_config.items():
        setattr(args, key, value)
    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    fold_idxes = separate_data_allfolds(graphs, args.seed)

    model_acc_avg = []
    for model_name in selected_model_list:
        setattr(args, 'model', model_name)
        model_config = model_config_dict[model_name]
        for key, value in model_config.items():
            setattr(args, key, value)

        if args.test:
            args.epochs = 3

        print(f"==> Argparser args: {args}")

        # Main training and validation process
        print('-'*100)
        print(f"==> Dataset {args.dataset} and Model {args.model} training and validation...")

        train_acc_per_fold, train_loss_per_fold, test_acc_per_fold = [], [], []

        for fold_idx in range(len(fold_idxes)):
            print('-'*100)
            print(f"===> Dataset {args.dataset} and Model {args.model} fold {fold_idx+1} training...")
            train_idx, test_idx = fold_idxes[fold_idx]

            train_graphs = [graphs[i] for i in train_idx]
            test_graphs = [graphs[i] for i in test_idx]

            model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            train_acc_per_epoch, train_loss_per_epoch, test_acc_per_epoch = [], [], []

            for epoch in range(1, args.epochs + 1):
                scheduler.step()

                train_loss = train(args, model, device, train_graphs, optimizer, epoch)
                train_acc, test_acc = test(args, model, device, train_graphs, test_graphs, epoch)

                train_acc_per_epoch.append(train_acc)
                train_loss_per_epoch.append(train_loss)
                test_acc_per_epoch.append(test_acc)

            train_acc_per_fold.append(train_acc_per_epoch)
            train_loss_per_fold.append(train_loss_per_epoch)
            test_acc_per_fold.append(test_acc_per_epoch)

            #print(model.eps)
            print()

        train_acc_per_epoch_folds, train_loss_per_epoch_folds, test_acc_per_epoch_folds = [], [], []
        for i in range(args.epochs):
            train_acc_this_epoch = []
            train_loss_this_epoch = []
            test_acc_this_epoch = []
            for j in range(len(fold_idxes)):
                train_acc_this_epoch.append(train_acc_per_fold[j][i])
                train_loss_this_epoch.append(train_loss_per_fold[j][i])
                test_acc_this_epoch.append(test_acc_per_fold[j][i])
            train_acc_per_epoch_folds.append(train_acc_this_epoch)
            train_loss_per_epoch_folds.append(train_loss_this_epoch)
            test_acc_per_epoch_folds.append(test_acc_this_epoch)

        train_acc_statistics, test_acc_statistics = [], []
        for i in range(args.epochs):
            train_acc_avg = st.mean(train_acc_per_epoch_folds[i])
            train_acc_std = st.stdev(train_acc_per_epoch_folds[i])
            train_loss_avg = st.mean(train_acc_per_epoch_folds[i])
            test_acc_avg = st.mean(test_acc_per_epoch_folds[i])
            test_acc_std = st.stdev(test_acc_per_epoch_folds[i])

            train_acc_statistics.append([train_acc_avg*100, train_acc_std*100])
            test_acc_statistics.append([test_acc_avg*100, test_acc_std*100])

            #writer.add_scalars(f"{args.model}_Train_Loss", train_loss_avg, i)
            writer.add_scalars(f"Acc/Train", {f"{args.model}": train_acc_avg}, i)
            writer.add_scalars(f"Acc/Test", {f"{args.model}": test_acc_avg}, i)

    
        max_idx = 0
        for i, (test_acc_avg, _) in enumerate(test_acc_statistics):
            if test_acc_avg > test_acc_statistics[max_idx][0]:
                max_idx = i

        print(f"==> Dataset {args.dataset} and Model {args.model}: train_acc: {train_acc_statistics[max_idx][0]:.1f} +- {train_acc_statistics[max_idx][1]:.1f} \
 		    test_acc: {test_acc_statistics[max_idx][0]:.1f} +- {test_acc_statistics[max_idx][1]:.1f}")
        model_acc_avg.append({f"{args.model}_train": train_acc_statistics[max_idx], f"{args.model}_test": test_acc_statistics[max_idx]})

	
    writer.close()
    print('*'*100)
    print(f"=> All Done.")
    print(f"=> {args.dataset} Summary:")
    for i, model in enumerate(selected_model_list):
        print(f"{model}: {model_acc_avg[i]}")

if __name__ == '__main__':
    main()
