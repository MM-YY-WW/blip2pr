import scanpy as sc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold

class ProteinExpressionDataset(Dataset):
    def __init__ (self, args, mode):
        self.dataset = args.dataset
        self.mode = mode
        if self.dataset == 'mye':
            self.train_X, self.test_X, self.train_y, self.test_y, self.unique_train_cell_types, self.unique_test_cell_types, self.gene_names = _load_mye(filepath="./dataset/mye/")
        elif self.dataset == 'pancreas':
            self.train_X, self.test_X, self.train_y, self.test_y, self.unique_train_cell_types, self.unique_test_cell_types, self.gene_names = _load_pancreas(filepath="./dataset/hPancreas/")
        elif self.dataset == 'ms':
            self.train_X, self.test_X, self.train_y, self.test_y, self.unique_train_cell_types, self.unique_test_cell_types, self.gene_names = _load_ms(filepath="./dataset/multiple_sclerosis/")
        elif self.dataset == 'zheng68k':
            self.train_X, self.test_X, self.train_y, self.test_y, self.unique_train_cell_types, self.unique_test_cell_types, self.gene_names = _load_zheng68k(args,filepath = "./dataset/zheng68K/")
        else:
            raise NotImplementedError('For now only support the following datasets [mye, pancreas, ms]')   
        self.gene_summary_embedding = _generate_gene_summary_embedding(gene_names=self.gene_names, args=args)

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_X) 
        elif self.mode == 'test':
            return len(self.test_X)
        elif self.mode == 'all':
            return len(self.X)
        
    def get_true_label(self):
        if self.mode == 'train':
            return self.unique_train_cell_types
        else:
            return self.unique_test_cell_types
    
    def get_gene_summary_embedding(self):
        #return names for all of the genes (3000) and the gene summary embedding, missing summary is set to all zeros
        return self.gene_names, self.gene_summary_embedding
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.train_X[index], self.train_y[index]
        else:
            return self.test_X[index], self.test_y[index]
        

def _load_mye(filepath):
    train = sc.read_h5ad(filename = filepath + "reference_adata.h5ad")
    test = sc.read_h5ad(filename = filepath + "query_adata.h5ad")
    train_X = train.X
    test_X = test.X
    train_df = pd.DataFrame(train.obs['cell_type'])
    train_df['CellTypeLabels'], unique_train_cell_types = pd.factorize(train_df['cell_type'])
    test_df = pd.DataFrame(test.obs['cell_type'])
    test_df['CellTypeLabels'], unique_test_cell_types = pd.factorize(test_df['cell_type'])
    assert train.var_names.tolist() == test.var_names.tolist()
    assert len(train.var_names.tolist()) == 3000
    gene_names = train.var_names.tolist() 
    return train_X, test_X, train_df['CellTypeLabels'], test_df['CellTypeLabels'], unique_train_cell_types, unique_test_cell_types, gene_names


def _load_pancreas(filepath):
    train = sc.read_h5ad(filename = filepath + "demo_train.h5ad")
    test = sc.read_h5ad(filename = filepath + "demo_test.h5ad")
    train_X = train.X
    test_X = test.X
    train_df = pd.DataFrame(train.obs['Celltype'])
    train_df['CellTypeLabels'], unique_train_cell_types = pd.factorize(train_df['Celltype'])
    test_df = pd.DataFrame(test.obs['Celltype'])
    test_df['CellTypeLabels'], unique_test_cell_types = pd.factorize(test_df['Celltype'])
    assert train.var_names.tolist() == test.var_names.tolist()
    assert len(train.var_names.tolist()) == 3000
    gene_names = train.var_names.tolist() 
    return train_X, test_X, train_df['CellTypeLabels'], test_df['CellTypeLabels'], unique_train_cell_types, unique_test_cell_types, gene_names

def _load_ms(filepath):
    train = sc.read_h5ad(filename = filepath + "filtered_ms_adata.h5ad") #[13468,3000]
    test = sc.read_h5ad(filename = filepath + "c_data.h5ad") # [7844, 3000]
    train_X = train.X.toarray()
    test_X = test.X.toarray()
    train_df = pd.DataFrame(train.obs['celltype'])
    train_df['CellTypeLabels'], unique_train_cell_types = pd.factorize(train_df['celltype'])
    test_df = pd.DataFrame(test.obs['celltype'])
    test_df['CellTypeLabels'], unique_test_cell_types = pd.factorize(test_df['celltype'])
    assert train.var['gene_name'].tolist() == test.var['gene_name'].tolist()
    assert len(train.var['gene_name'].tolist()) == 3000
    gene_names = train.var['gene_name'].tolist() 

    return train_X, test_X, train_df['CellTypeLabels'], test_df['CellTypeLabels'], unique_train_cell_types, unique_test_cell_types, gene_names

def _load_zheng68k(args, filepath):
    data = sc.read_h5ad(filename=filepath + "Zheng68K.h5ad")
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    X = data.X.toarray()
    data_df = pd.DataFrame(data.obs['celltype'])
    data_df['CellTypeLabels'], unique_cell_types = pd.factorize(data_df['celltype'])
    gene_names = data.var_names.tolist() #16906
    y = data_df['CellTypeLabels']
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size = 0.8, random_state = args.seed)
    unique_train_cell_types = [unique_cell_types.tolist()[i] for i in train_y.unique().tolist()]
    unique_test_cell_types = [unique_cell_types.tolist()[i] for i in test_y.unique().tolist()]
    #train_X_tensor = torch.from_numpy(train_X).long()
    #test_X_tensor = torch.from_numpy(test_X).long()
    return train_X, test_X, train_y, test_y, unique_train_cell_types, unique_test_cell_types, gene_names

def _load_aorta(args, filepath):
    return 0
def _load_heart(args, filepath):
    return 0

def _generate_gene_summary_embedding(gene_names, args):
    with open(args.gene_summary_path, 'rb') as pf:
        all_gene_summary_embedding = pkl.load(pf)
    with open(args.gene_summary_manual_path, 'rb') as pf:
        manual_summary_embedding = pkl.load(pf)
    gene_summary_embedding = []
    for i in range(len(gene_names)):
        if gene_names[i] in all_gene_summary_embedding:
            gene_summary_embedding.append(all_gene_summary_embedding[gene_names[i]].tolist())
        else:
            #1536
            if args.missing_summary == "zeros":
                embedding_length = len(all_gene_summary_embedding[list(all_gene_summary_embedding.keys())[0]])
                gene_summary_embedding.append([0]*embedding_length)
            elif args.missing_summary == "ones":
                embedding_length = len(all_gene_summary_embedding[list(all_gene_summary_embedding.keys())[0]])
                gene_summary_embedding.append([1]*embedding_length)
            elif args.missing_summary == "auto":
                # to be implemented reuqest openai api to generate summary and embedding
                raise NotImplementedError()
            elif args.missing_summary == "manual":
                gene_summary_embedding.append(manual_summary_embedding[gene_names[i]][0].tolist())
            elif args.missing_summary == "random":
                embedding_length = len(all_gene_summary_embedding[list(all_gene_summary_embedding.keys())[0]])
                gene_summary_embedding.append(torch.randn(embedding_length).tolist())

            else:
                raise KeyError("args.missing_summary should be chosen from [zeros, ones, auto, manual]")
    return gene_summary_embedding

#_load_zheng68k(filepath = "./dataset/zheng68K/")
# _,_,_,_,_,_,mye_gene_name=_load_mye(filepath="./dataset/mye/")
#_generate_gene_summary_embedding(gene_names=mye_gene_name, file_path='gene_summary/data_embedding/GPT_3_5_gene_embeddings.pickle')
# _load_pancreas(filepath="./dataset/hPancreas/")
# _load_ms(filepath="./dataset/multiple_sclerosis/")

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data # [54760, 16906]
        self.label = label
        a=1

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2 # greater than 5 expression level = 5
        full_seq = torch.from_numpy(full_seq).long() #convert to tensor
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device) #[16907]? why add extra zero at the end?
        seq_label = self.label[rand_start] #like tensor(8)
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]