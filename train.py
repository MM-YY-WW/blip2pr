from dataset import ProteinExpressionDataset, SCDataset
from torch.utils.data import DataLoader
import scanpy as sc
import torch
import pickle as pkl
import numpy as np
import pandas as pd
import math
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
import torch.nn as nn
import random 
from model import *
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging
import umap
from PIL import Image, ImageDraw
import io
from performer_pytorch import PerformerLM


CUDA_LAUNCH_BLOCKING = 1

def calculate_metric(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    return accuracy, f1, precision, recall

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministric = True
        torch.backends.cudnn.benchmark = False

def umap_visualization(embedding, labels, path=''):
    reducer = umap.UMAP()
    umap_embeddings = reducer.fit_transform(embedding)
    plt.figure(figsize=(10, 6))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar()
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    #path='./gif/temp/' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.png'
    plt.savefig(path)
    return path

def result_gif(train_frame_paths, test_frame_paths, gif_path):
    if len(train_frame_paths) != len(test_frame_paths):
        raise ValueError("train_frames and test_frames must be of the same length")
    combined_frames = []
    for train_path, test_path in zip(train_frame_paths, test_frame_paths):
        with Image.open(train_path) as train_frame, Image.open(test_path) as test_frame:
            width, height = train_frame.size
            combined_image = Image.new('RGB', (2 * width, height), (255, 255, 255))  
            combined_image.paste(train_frame, (0, 0))
            combined_image.paste(test_frame, (width, 0))
            combined_frames.append(combined_image.copy())
    combined_frames[0].save(gif_path, save_all=True, append_images=combined_frames[1:], duration=500, loop=0)

def plot_results(train_acc_list, test_acc_list, train_precision_list, test_precision_list, train_f1_list, test_f1_list, train_recall_list, test_recall_list, args):
    # Create a directory to store the plots
    if not os.path.exists("./plot_results"):
        os.makedirs("./plot_results")

    # Generate a unique timestamp for the plot filename
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # Set up the figure and subplots
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"lr = {args.learning_rate}, epoch = {args.epoch}, dataset = {args.dataset}, seed = {args.seed}, batch_size = {args.batch_size}, model_name = {args.model_name}, hidden_k_size = {args.hidden_k_size}, missing_summary = {args.missing_summary}, scbert_out_model = {args.scbert_out_model}")
    logging.info(f"lr = {args.learning_rate}, epoch = {args.epoch}, dataset = {args.dataset}, seed = {args.seed}, batch_size = {args.batch_size}, model_name = {args.model_name}, hidden_k_size = {args.hidden_k_size}, missing_summary = {args.missing_summary}, scbert_out_model = {args.scbert_out_model}")
    # Plot training accuracy
    axs[0, 0].plot(train_acc_list, label='Train Accuracy')
    axs[0, 0].set_title('Train Accuracy')

    # Plot testing accuracy
    axs[1, 0].plot(test_acc_list, label='Test Accuracy')
    axs[1, 0].set_title('Test Accuracy')

    # Plot training precision
    axs[0, 1].plot(train_precision_list, label='Train Precision')
    axs[0, 1].set_title('Train Precision')

    # Plot testing precision
    axs[1, 1].plot(test_precision_list, label='Test Precision')
    axs[1, 1].set_title('Test Precision')

    # Plot training F1 score
    axs[0, 2].plot(train_f1_list, label='Train F1 Score')
    axs[0, 2].set_title('Train F1 Score')

    # Plot testing F1 score
    axs[1, 2].plot(test_f1_list, label='Test F1 Score')
    axs[1, 2].set_title('Test F1 Score')

    # Plot training recall
    axs[0, 3].plot(train_recall_list, label='Train Recall')
    axs[0, 3].set_title('Train Recall')

    # Plot testing recall
    axs[1, 3].plot(test_recall_list, label='Test Recall')
    axs[1, 3].set_title('Test Recall')

    # Add legends
    axs[0, 0].legend()
    axs[1, 0].legend()
    axs[0, 1].legend()
    axs[1, 1].legend()
    axs[0, 2].legend()
    axs[1, 2].legend()
    axs[0, 3].legend()
    axs[1, 3].legend()

    plot_filename = f"./plot_results/{timestamp}_{args.dataset}_{args.model_name}.png"
    logging.info(plot_filename)
    plt.savefig(plot_filename)
    plt.show()


def train(args, model, device, dataloader, optimizer, loss_fn, gene_summary_embedding):
    model.train()
    total_loss = 0
    y_list = []
    pred_list = []
    all_embedding = []
    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        X = batch[0]
        y = batch[1]
        X,y,gene_summary_embedding = X.to(device), y.to(device), gene_summary_embedding.to(device)
        if args.model_name == 'scBERT':
            zeros = torch.zeros(X.shape[0], 1, device = device)
            X = torch.cat((X, zeros), dim=1)
            X[X>5] = 5
            X = X.long()
            out, pred = model(X)
        else:
            out, pred = model(X, gene_summary_embedding)

        if args.gif == 1:
            all_embedding.append(out)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred_list += [np.argmax(i) for i in pred.cpu().detach().numpy()]
        y_list += y.tolist()
    if args.gif == 1:
        all_embedding = torch.cat(all_embedding, dim=0)
    accuracy, f1, precision, recall =  calculate_metric(y_list,pred_list)
    return total_loss/step, accuracy, f1, precision, recall, all_embedding

def test(args, model, device, dataloader, loss_fn, gene_summary_embedding):
    model.eval()
    total_loss = 0
    y_list = []
    pred_list = []
    all_embedding = []
    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        X = batch[0]
        y = batch[1]
        X,y,gene_summary_embedding = X.to(device), y.to(device), gene_summary_embedding.to(device)
        if args.model_name == 'scBERT':
            zeros = torch.zeros(X.shape[0], 1, device = device)
            X = torch.cat((X, zeros), dim=1)
            X[X>5] = 5
            X = X.long()
            out, pred = model(X)
        else:
            out, pred = model(X, gene_summary_embedding)
        if args.gif == 1:
            all_embedding.append(out)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        pred_list += [np.argmax(i) for i in pred.cpu().detach().numpy()]
        y_list += y.tolist()
    if args.gif == 1:
        all_embedding = torch.cat(all_embedding, dim=0)
    accuracy, f1, precision, recall =  calculate_metric(y_list,pred_list)
    return total_loss/step, accuracy, f1, precision, recall, all_embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='zheng68k', help="choose the dataset used to finetune from [mye, ms, pancreas, zheng68k]")
    parser.add_argument("--epoch", type=int, default = 100)
    parser.add_argument("--learning_rate", type=float, default = 1e-4)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str, default='scBERT',
                        help="which model to use for finetuning \
                        'GenePT_Exponly': only the data.X gene expression level data \
                        'GenePT_W': the gene_summary_embedding weighted by the expression level \
                        'GenePT_attention': X K Kt gene_summary_embedding\
                        'scBERT': use scBERT to generate gene embedding rather than just use original expression level")
    parser.add_argument("--model_path", type=str, default = './pretrained_model/', help="the folder path to load pretrain model")
    parser.add_argument("--device", type=int, default=4, help="which gpu to choose")
    parser.add_argument("--gene_summary_path", type=str, default = './gene_summary/data_embedding/GPT_3_5_gene_embeddings.pickle')
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--hidden_k_size", type=int, default = 512, help = "the dimension of the k matrix in GenePT_attention model")
    parser.add_argument("--missing_summary", type=str, default= 'ones', 
                        help = "method of dealing with the missing gene summary embedding. \
                        'zeros': use all zeros embedding,\
                        'ones' use all ones embedding, \
                        'auto': request openai api to automatically generate missing embedding, \
                        'manual': use predefined summary \
                        'random': use random generated embedding from uniform distribution" )
    parser.add_argument("--scbert_out_model", type=str, default = 'GenePT_attention')
    parser.add_argument("--gene_summary_manual_path", type=str, default = "./gene_summary/manual/GPT_3_5_gene_embeddings_generated.pickle")
    parser.add_argument("--log_path", type=str, default = "./log/", help="log folder")
    parser.add_argument("--plot", type=int, default=1, help='0 for not plot results, 1 plot results of acc, precision, f1, recall')
    parser.add_argument("--gif", type=int, default=0, help="generate the gif of umap visualization or not")
    parser.add_argument("--gif_frequency", type=int, default=10, help="for every gif_frequency epoches generate one figure")
    args = parser.parse_args()
    set_seed(args.seed)

    logging.basicConfig(filename=args.log_path+args.dataset+'.log', level = logging.INFO)
    train_dataset = ProteinExpressionDataset(args, mode='train')
    test_dataset = ProteinExpressionDataset(args, mode='test')
    train_true_label = train_dataset.get_true_label()
    test_true_label = test_dataset.get_true_label()
    gene_names, gene_summary_embedding = train_dataset.get_gene_summary_embedding()
    gene_summary_embedding = torch.as_tensor(gene_summary_embedding, device = device)
    train_loader = DataLoader(train_dataset,  batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset,  batch_size=args.batch_size)
    num_of_gene = len(train_loader.dataset[0][0])
    #get embeddings for umap visualization and gif generation
    init_train_embedding = []
    all_train_y = []
    init_test_embedding = []
    all_test_y = []
    for batch in train_loader:
        init_train_embedding.append(batch[0])
        all_train_y.append(batch[1])
    for batch in test_loader:
        init_test_embedding.append(batch[0])
        all_test_y.append(batch[1])
    init_train_embedding = torch.cat(init_train_embedding, dim=0)
    all_train_y = torch.cat(all_train_y, dim=0)
    init_test_embedding = torch.cat(init_test_embedding, dim=0)
    all_test_y = torch.cat(all_test_y, dim=0)

    # define models
    if args.model_name == "GenePT_W":   
        model = GenePT_W(num_class = len(train_true_label))
    elif args.model_name == "GenePT_Exponly":
        model = GenePT_Exponly(num_class = len(train_true_label), num_gene=num_of_gene)
    elif args.model_name == "GenePT_attention":
        model = GenePT_attention(num_class=len(train_true_label), hidden_k_size=args.hidden_k_size, num_gene=num_of_gene)
    elif args.model_name == "scBERT":
        model = PerformerLMM(
            num_tokens = 7,
            dim = 200,
            depth = 6,
            max_seq_len = num_of_gene+1,
            heads = 10,
            local_attn_heads = 0,
            # use gene2vec encoding or not default in scBERT args is True so True here
            g2v_position_emb = True
        )
        model_path = args.model_path + args.model_name + '/panglao_pretrain.pth'
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['model_state_dict'])
        for param in model.parameters():
            param.requires_grad = False
        for param in model.norm.parameters():
            param.requires_grad = True
        for param in model.performer.net.layers[-2].parameters():
            param.requires_grad = True
        #model.to_out = scBERT_Identity(dropout = 0., h_dim=128, out_dim = len(train_true_label), num_gene=num_of_gene+1)
        if args.scbert_out_model == 'GenePT_Exponly':
            model.to_out = GenePT_Exponly(num_class=len(train_true_label), num_gene=num_of_gene+1)
        elif args.scbert_out_model == 'GenePT_W':
            model.to_out = GenePT_W(num_class=len(train_true_label), text_embedding = gene_summary_embedding)
        elif args.scbert_out_model == 'GenePT_attention':
            model.to_out = GenePT_attention(num_class=len(train_true_label),hidden_k_size=args.hidden_k_size, num_gene=num_of_gene, text_embedding = gene_summary_embedding)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.decay)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")
    model = model.to(device)
    train_acc_list = []
    test_acc_list = []
    train_precision_list = []
    test_precision_list = []
    train_f1_list =[]
    test_f1_list = []
    train_recall_list = []
    test_recall_list = []
    #umap_visualization(embedding=torch.matmul(init_train_embedding, gene_summary_embedding), labels=all_train_y, path = './gif/GenePT_W/mye/mye_train_manual.png')
    #umap_visualization(embedding=torch.matmul(init_test_embedding, gene_summary_embedding), labels=all_test_y, path = './gif/GenePT_W/mye/mye_test_manual.png')
    if args.gif == 1:
        train_frames = []
        test_frames = []
        train_frames.append(umap_visualization(embedding=init_train_embedding, labels = all_train_y))
        test_frames.append(umap_visualization(embedding = init_test_embedding, labels = all_test_y))
    for i in range(1, args.epoch+1):
        print("====Epoch" + str(i))
        train_loss, train_acc, train_f1, train_precision, train_recall, train_embedding = train(args=args, model=model, device=device, dataloader=train_loader, optimizer=optimizer, loss_fn = loss_fn, gene_summary_embedding=gene_summary_embedding)
        test_loss, test_acc, test_f1, test_precision, test_recall, test_embedding = test(args=args, model=model, device=device, dataloader=test_loader,loss_fn=loss_fn,gene_summary_embedding=gene_summary_embedding)
        print(f"Train Loss {train_loss}, Train Acc {train_acc}")
        print(f"Test Loss {test_loss}, Test Acc {test_acc}")
        if args.gif == 1 and i % args.gif_frequency == 0:
            train_frames.append(umap_visualization(embedding=train_embedding, labels = all_train_y))
            test_frames.append(umap_visualization(embedding = test_embedding, labels = all_test_y))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_precision_list.append(train_precision)
        test_precision_list.append(test_precision)
        train_f1_list.append(train_f1)
        test_f1_list.append(test_f1)
        train_recall_list.append(train_recall)
        test_recall_list.append(test_recall)

    max_index = test_acc_list.index(max(test_acc_list))
    logging.info(f'Best Test results at Epoch{max_index}, Acc: {test_acc_list[max_index]}, F1: {test_f1_list[max_index]}, Precision: {test_precision_list[max_index]}, Recall: {test_recall_list[max_index]}')
    if args.gif == 1:
        result_gif(train_frame_paths = train_frames, test_frame_paths=test_frames, gif_path = './gif/'+args.dataset+ '_' + args.model_name + '.gif')
    
    if args.plot == 1:
        plot_results(train_acc_list=train_acc_list, 
                    test_acc_list = test_acc_list,
                    train_precision_list=train_precision_list,
                    test_precision_list=test_precision_list,
                    train_f1_list=train_f1_list,
                    test_f1_list = test_f1_list,
                    train_recall_list = train_recall_list,
                    test_recall_list = test_recall_list,
                    args = args)

if __name__ =="__main__":
    main()







