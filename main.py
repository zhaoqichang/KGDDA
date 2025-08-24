import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import argparse

from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from split_data import split_k_fold
from ampligraph.latent_features import ComplEx,TransE,DistMult,HolE,TransE,TransE,ConvE,ConvKB
from sklearn import metrics
from ampligraph.utils import save_model,restore_model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam,Adagrad,Adamax
from sklearn.decomposition import PCA
from tensorflow import keras
from DNN import generate_model
from DNN import classification_accuracy
from torch.utils.data import Dataset
import torch
from load_log import make_print_to_file
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



parser = argparse.ArgumentParser(description='Ours')

parser.add_argument('--data-name', default='MSI_new',
                    help='dataset name')
parser.add_argument('--split', default='warm',
                    help='split data')
parser.add_argument('--seed', default=42,
                    help='seed')
parser.add_argument('--KGEM', default='ComplEx',
                    help='dataset name')
parser.add_argument('--mode', default='KE_DNN_4',
                    help='test_dda')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')


parser.add_argument('--dropout_n', type=float, default=0.4,
                    help='random drops neural node with this prob')
parser.add_argument('--dropout_e', type=float, default=0.1,
                    help='random drops edge with this prob')
parser.add_argument('--valid_interval', type=int, default=1)
parser.add_argument('--hop', type=int, default=2,
                    help='the number of neighbor (default: 2)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--force-undirected', action='store_true', default=False,
                    help='in edge dropout, force (x, y) and (y, x) to be dropped together')
parser.add_argument('--patient', type=int, default=20,
                    help='zaoting')


parser.add_argument('--valid_auc', type=float, default=0.91, metavar='LR',
                    help='threshold value')

parser.add_argument('--patient_epoch', type=int, default=40,
                    help='when start ')

parser.add_argument('--drdi_feature', type=str,default='one',
                    help='drug disease feature')


parser.add_argument('--nhead', type=int, default=8,
                    help='几头')
parser.add_argument('--num_layers', type=int, default=2,
                    help='层')
parser.add_argument('--dim_feedforward', type=int, default=256,
                    help='')

args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

print(args)
make_print_to_file(args,path=f'../log/{args.data_name}/')


print('HI')


if args.data_name=='MSI_new':
    relation = 'Drug_Disease'
    if args.KGEM=='DistMult':
        batch_count, lr, lam = 1000, 0.0001, 0.001
    elif args.KGEM=='ComplEx':
        batch_count, lr, lam = 2000, 0.0001, 0.001
    elif args.KGEM=='TransE':
        batch_count, lr, lam = 2000, 0.0001, 0.001
    elif args.KGEM=='ConvKB':
        batch_count, lr, lam = 1000, 0.0001, 0.001
elif args.data_name=='My':
    relation = 'drug_disease'
    if args.KGEM == 'DistMult':
        batch_count, lr, lam = 5000, 0.0001, 0.001
    elif args.KGEM == 'ComplEx':
        batch_count, lr, lam = 5000, 0.0001, 0.001
    elif args.KGEM=='TransE':
        batch_count, lr, lam = 5000, 0.01, 0.0001
    elif args.KGEM=='ConvKB':
        batch_count, lr, lam = 5000, 0.001, 0.0001
elif args.data_name == 'DrugRep-KG':
    relation = 'Drug_Disease'
    if args.KGEM == 'ComplEx':
        batch_count, lr, lam = 1000, 0.0001, 0.001


print(args.data_name,args.split,args.seed,args.KGEM,args.mode,args.batch_size,args.drdi_feature,args.valid_auc,args.patient_epoch)


kg = pd.read_csv(f'../data/{args.data_name}/kg.csv', dtype=str)

dda = kg[kg['relation']==relation]

dda.reset_index(drop=True, inplace=True)
kg = pd.concat([kg,dda], axis=0, ignore_index=True)
kg = kg.drop_duplicates(keep=False)
kg.reset_index(drop=True, inplace=True)

nodetype_df = pd.read_csv(f'../data/{args.data_name}/nodetype_bf.tsv', sep='\t', dtype=str)
entities = nodetype_df['node'].values

head_le = LabelEncoder()
tail_le = LabelEncoder()
head_le.fit(list(dda['head']))
tail_le.fit(list(dda['tail']))
mms = MinMaxScaler(feature_range=(0,1))

def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

def load_data_KE():
    train = {}
    valid = {}
    test = {}
    if os.path.exists(f'../../dataset/{args.data_name}/data_fold/{args.split}/fold_{0}_{args.seed}_train.csv'):
        for i in range(10):
            train[i] = pd.read_csv(f'../../dataset/{args.data_name}/data_fold/{args.split}/fold_{i}_{args.seed}_train.csv')
            valid[i] = pd.read_csv(f'../../dataset/{args.data_name}/data_fold/{args.split}/fold_{i}_{args.seed}_valid.csv')
            test[i] = pd.read_csv(f'../../dataset/{args.data_name}/data_fold/{args.split}/fold_{i}_{args.seed}_test.csv')
    else:
        train,valid,test = split_k_fold(args.data_name, args.seed, args.split)

    for i in range(10):
        train[i]['relation'] = [relation] * len(train[i])
        train[i] = train[i][['drug',  'relation','indication','label']]
        train[i].columns = ['head', 'relation', 'tail','label']
        valid[i]['relation'] = [relation] * len(valid[i])
        valid[i] = valid[i][['drug', 'relation', 'indication', 'label']]
        valid[i].columns = ['head', 'relation', 'tail', 'label']

        test[i]['relation'] = [relation] * len(test[i])
        test[i] = test[i][['drug', 'relation', 'indication', 'label']]
        test[i].columns = ['head', 'relation', 'tail', 'label']

    return train,valid,test


def get_embeddings_3(model,data,drug_f,disease_f):
    sub_embeddings = np.array([model.get_embeddings([data['head'][i]], embedding_type='entity') for i in range(len(data))])
    sub_embeddings = np.squeeze(sub_embeddings, axis=1)
    obj_embeddings = np.array([model.get_embeddings([data['tail'][i]], embedding_type='entity') for i in range(len(data))])
    obj_embeddings = np.squeeze(obj_embeddings, axis=1)

    drug_embeddings =np.array([drug_f[data['head'][i]] for i in range(len(data))])
    disease_embeddings =np.array([disease_f[data['tail'][i]] for i in range(len(data))])

    return sub_embeddings,obj_embeddings,drug_embeddings,disease_embeddings

class InteractionDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

        self.label = torch.LongTensor(self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        idx = idx % len(self)

        X = self.data[idx]
        X = torch.from_numpy(X).float()
        label = self.label[idx]
        return X, label



def train_KE_2(i,train,valid,test):
    columns = ['head', 'relation', 'tail']
    if os.path.exists(f'../model/{args.data_name}/KGEM_{args.mode}_{args.KGEM}_warm_{i}.pkl'):
        model = restore_model(
            model_name_path= f'../model/{args.data_name}/KGEM_{args.mode}_{args.KGEM}_warm_{i}.pkl')
    else:

        #model = restore_model(model_name_path=f'../model/{args.data_name}/KGEM_initail_{args.KGEM}.pkl')
        model = ComplEx(batches_count=batch_count,
                seed=0,
                epochs=1000,
                k=400,
                # embedding_model_params={'corrupt_sides':'o'},
                optimizer='adam',
                optimizer_params={'lr': lr},
                loss='pairwise',  # pairwise
                regularizer='LP',
                regularizer_params={'p': 3, 'lambda': lam},
                verbose=True)

        train_pos = train[train['label'] == 1]
        train_pos.reset_index(drop=True, inplace=True)

        train_pos = train_pos[columns]

        data = pd.concat([kg, train_pos], axis=0).reset_index(drop=True)

        print('fit')
        model.fit(data.values, early_stopping=True, early_stopping_params=
        {
            'x_valid': train_pos.values,
            # validation set, here we use training set for validation
            'criteria': 'mrr',  # Uses mrr criteria for early stopping
            'burn_in': 10,  # early stopping kicks in after 10 epochs
            'check_interval': 2,  # validates every 2th epoch
            'stop_interval': 10,  # stops if 3 successive validation checks are bad.
            'x_filter': dda.values,  # Use filter for filtering out positives
            'corrupt_side': 'o'  # corrupt object (but not at once)
        })
        save_model(model, model_name_path=f'../model/{args.data_name}/KGEM_{args.mode}_{args.KGEM}_{args.split}_{i}.pkl')

    print('test')
    #test = pd.concat([test, valid], axis=0).reset_index(drop=True)
    test_score = model.predict(test[columns])
    test_label = test['label'].values
    auc = roc_auc(test_label, test_score)
    aupr = pr_auc(test_label, test_score)

    np.savetxt(f'../result/{args.data_name}/{args.KGEM}_scores{i}.txt', test_score, delimiter=" ")
    np.savetxt(f'../result/{args.data_name}/{args.KGEM}_labels{i}.txt', test_label, delimiter=" ")
    return auc, aupr


def train_KE_all_PCA(i,train,valid,test,drug_f,disease_f):

    #KE_model = restore_model(model_name_path= f'../model/{args.data_name}/KGEM_KE_2_{args.KGEM}_warm_{i}.pkl')

    KE_model = restore_model(model_name_path=f'../model/{args.data_name}/KGEM_{args.KGEM}_{args.split}_{i}.pkl')

    train_dr_1,train_di_1,train_dr_2,train_di_2 = get_embeddings_3(KE_model, train, drug_f, disease_f)
    train_label = train['label'].values

    valid_dr_1,valid_di_1,valid_dr_2,valid_di_2 = get_embeddings_3(KE_model, valid, drug_f, disease_f)
    test_label = test['label'].values

    test_dr_1,test_di_1,test_dr_2,test_di_2 = get_embeddings_3(KE_model, test, drug_f, disease_f)
    valid_label = valid['label'].values


    pca_dr = PCA(n_components=400)
    train_dr_2 = pca_dr.fit_transform(train_dr_2)
    valid_dr_2 = pca_dr.transform(valid_dr_2)
    test_dr_2 = pca_dr.transform(test_dr_2)

    pca_di = PCA(n_components=400)
    train_di_2 = pca_di.fit_transform(train_di_2)
    valid_di_2 = pca_di.transform(valid_di_2)
    test_di_2 = pca_di.transform(test_di_2)

    pca_sub = PCA(n_components=400)
    train_dr_1 = pca_sub.fit_transform(train_dr_1)
    valid_dr_1 = pca_sub.transform(valid_dr_1)
    test_dr_1 = pca_sub.transform(test_dr_1)

    pca_obj = PCA(n_components=400)
    train_di_1 = pca_obj.fit_transform(train_di_1)
    valid_di_1 = pca_obj.transform(valid_di_1)
    test_di_1 = pca_obj.transform(test_di_1)


    train_feats = np.concatenate([train_dr_1, train_dr_2, train_di_1, train_di_2], axis=1)
    valid_feats = np.concatenate([valid_dr_1, valid_dr_2, valid_di_1, valid_di_2], axis=1)
    test_feats = np.concatenate([test_dr_1, test_dr_2, test_di_1, test_di_2], axis=1)

    train_dataset = InteractionDataset(train_feats, train_label)
    valid_dataset = InteractionDataset(valid_feats,valid_label)
    test_dataset = InteractionDataset(test_feats, test_label)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    flag = 0
    if os.path.exists(f'../model/{args.data_name}/KGEM_{args.mode}_{args.KGEM}_{args.split}_{args.valid_auc}_{args.patient_epoch}_{args.drdi_feature}_{args.nhead}_{args.num_layers}_{args.dim_feedforward}_{i}.pkl'):
        KE_DNN = torch.load(f'../model/{args.data_name}/KGEM_{args.mode}_{args.KGEM}_{args.split}_{args.valid_auc}_{args.patient_epoch}_{args.drdi_feature}_{args.nhead}_{args.num_layers}_{args.dim_feedforward}_{i}.pkl')
    else:
        KE_DNN,flag = generate_model(i,args,train_loader, valid_loader,train_feats.shape[1])
        torch.save(KE_DNN, f'../model/{args.data_name}/KGEM_{args.mode}_{args.KGEM}_{args.split}_{args.valid_auc}_{args.patient_epoch}_{args.drdi_feature}_{args.nhead}_{args.num_layers}_{args.dim_feedforward}_{i}.pkl')

    test_loss, test_acc, test_auc, test_aupr = classification_accuracy(KE_DNN, test_loader,train_feats.shape[1])

    return test_auc, test_aupr ,flag


if __name__ == '__main__':

    print('main')
    print(args.mode)
    result = pd.DataFrame(columns=['nhead','num_layers' ,'drdi_feature', 'valid_auc', 'patient_epoch', 'auc', 'aupr','epoch'])


    drug_features = pd.read_csv(f'../data/{args.data_name}/drug_feature.csv', index_col=0)
    disease_features = pd.read_csv(f'../data/{args.data_name}/disease_feature.csv', index_col=0)

    drug_f = {ind: np.array(drug_features.loc[ind]) for ind in drug_features.index}
    disease_f = {ind: np.array(disease_features.loc[ind]) for ind in disease_features.index}

    if args.mode =='KE_2':
        df = pd.DataFrame(columns=['fold', 'auc', 'aupr'])
        train, valid, test = load_data_KE()
        for i in range(10):
            print('*****************',i,'*****************')
            auc,aupr = train_KE_2(i,train[i],valid[i],test[i])
            df.loc[len(df.index)] = [i, auc, aupr]
            print(i, auc, aupr)
            df.to_csv(f'../result/{args.data_name}/KGEM_{args.mode}_{args.KGEM}_{args.split}.csv')


    elif args.mode == 'TE' :
        print(args.mode)
        train, valid, test = load_data_KE()
        valid_aucs = [0.]
        patient_epochs = [40]
        for valid_auc in valid_aucs:
            for patient_epoch in patient_epochs:
                df = pd.DataFrame(columns=['fold', 'auc', 'aupr', 'flag'])
                args.valid_auc = valid_auc
                args.patient_epoch = patient_epoch
                print('args.valid_auc,args.patient_epoch ', args.valid_auc, args.patient_epoch)
                for i in range(10):
                    print('*****************', i, '*****************')
                    auc, aupr, flag = train_KE_all_PCA(i, train[i], valid[i], test[i], drug_f, disease_f)
                    df.loc[len(df.index)] = [i, auc, aupr, flag]
                    print(i, auc, aupr, flag)
                    if args.mode =='KE_all_PCA':
                        df.to_csv(f'../result/{args.data_name}/KGEM_{args.mode}_{args.KGEM}_{args.split}_{args.valid_auc}_{args.patient_epoch}_{args.drdi_feature}.csv')
                    elif args.mode =='TE' or args.mode =='TE_2' or args.mode =='TE_3'or args.mode =='TE_4':
                        df.to_csv(
                            f'../result/{args.data_name}/KGEM_{args.mode}_{args.KGEM}_{args.split}_{args.valid_auc}_{args.patient_epoch}_{args.drdi_feature}_{args.nhead}_{args.num_layers}_{args.dim_feedforward}.csv')
                avg_auc = df['auc'].mean()
                avg_aupr = df['aupr'].mean()
                avg_epoch = df['flag'].mean()
                result.loc[len(result.index)] = [args.nhead,args.num_layers ,args.drdi_feature, valid_auc, patient_epoch, avg_auc, avg_aupr,avg_epoch]
                result.to_csv(f'../result/{args.data_name}/KGEM_{args.mode}_{args.split}_{args.valid_auc}.csv')
