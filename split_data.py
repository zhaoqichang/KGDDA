
import os
import scipy.sparse as ssp
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import warnings
#from argss import *
warnings.filterwarnings("ignore")


def data_split(pos_pairs,neg_pairs,seed,split,drug_df,disease_df,data_name):
    train = {}
    valid = {}
    test = {}
    k = 5
    if os.path.exists(f'../../dataset/{data_name}/data_fold/{split}/fold_{0}_{seed}_train.csv'):
        for i in range(k):
            print(k)
            train[i] = pd.read_csv(f'../../dataset/{data_name}/data_fold/{split}/fold_{i}_{seed}_train.csv')
            valid[i] = pd.read_csv(f'../../dataset/{data_name}/data_fold/{split}/fold_{i}_{seed}_valid.csv')
            test[i] = pd.read_csv(f'../../dataset/{data_name}/data_fold/{split}/fold_{i}_{seed}_test.csv')
        return train, valid, test

    pos_pairs = pos_pairs.sample(frac=1,random_state=0).reset_index(drop=True)
    neg_pairs = neg_pairs.sample(frac=1, random_state=1).reset_index(drop=True)
    drug_df = drug_df.sample(frac=1, random_state=2).reset_index(drop=True)
    disease_df = disease_df.sample(frac=1, random_state=3).reset_index(drop=True)

    pos_train ,pos_valid,pos_test = warm_split(pos_pairs,seed)
    neg_train,neg_valid,neg_test = warm_split(neg_pairs,seed)


    for i in range(k):
        train[i] = pd.concat([pos_train[i],neg_train[i]])
        valid[i] = pd.concat([pos_valid[i],neg_valid[i]])
        test[i] = pd.concat([pos_test[i],neg_test[i]])

        train[i] = train[i].sample(frac=1,random_state=5).reset_index(drop=True)
        valid[i] = valid[i].sample(frac=1,random_state=6).reset_index(drop=True)
        test[i] = test[i].sample(frac=1,random_state=7).reset_index(drop=True)

        h =None
        train[i].to_csv(f'../../dataset/{data_name}/data_fold/{split}/fold_{i}_{seed}_train.csv', index=False, header=True)
        valid[i].to_csv(f'../../dataset/{data_name}/data_fold/{split}/fold_{i}_{seed}_valid.csv', index=False, header=True)
        test[i].to_csv(f'../../dataset/{data_name}/data_fold/{split}/fold_{i}_{seed}_test.csv', index=False, header=True)

    return train,valid,test



def warm_split(df_pairs,seed):
    k = 5
    fold_size = len(df_pairs) // k
    train ={}
    valid = {}
    test = {}
    for i in range(k):
        #print(i)
        test_start = i * fold_size
        if i != k - 1 and i != 0:
            test_end = (i + 1) * fold_size
            testset = df_pairs[test_start:test_end]
            tvset = pd.concat([df_pairs[0:test_start], df_pairs[test_end:]])
        elif i == 0:
            test_end = fold_size
            testset = df_pairs[test_start:test_end]
            tvset = df_pairs[test_end:]
        else:
            testset = df_pairs[test_start:]
            tvset = df_pairs[0:test_start]

        # split training-set and valid-set
        trainset, validset = train_test_split(tvset, test_size=1.0/(k-1), random_state=seed)
        #print(f'train:{len(trainset)}, valid:{len(validset)}, test:{len(testset)}')
        train[i] =  trainset
        valid[i] = validset
        test [i] = testset
    return  train,valid,test


def generate_mine_neg(mtx_drdi,pos_pairs):
    num = len(pos_pairs)
    #print(len(pos_pairs))

    np.random.seed(100)
    np.random.shuffle(pos_pairs)

    flag = np.zeros(num)

    neg = np.empty((0, 2), dtype=int)
    for i in range(num):
        u_1, v_1 = pos_pairs[i]
        if flag[i] == 1:
            continue
        zero_indices = np.where(flag == 0)[0]
        j = -1
        for k in zero_indices:
            u, v = pos_pairs[k]
            if u_1 != u and v_1 != v:
                j = k
                break
        if j == -1:
            continue
        flag[i] = 1
        flag[j] = 1
        u_2, v_2 = pos_pairs[j]

        mtx_drdi[u_1, v_2] = -1
        mtx_drdi[u_2, v_1] = -1
        neg = np.vstack((neg, np.array([u_1, v_2])))
        neg = np.vstack((neg, np.array([u_2, v_1])))

    row_pos = np.count_nonzero(mtx_drdi==1  ,axis=1)
    row_neg = np.count_nonzero(mtx_drdi==-1,axis=1)

    col_pos = np.count_nonzero(mtx_drdi==1,axis=0)
    col_neg = np.count_nonzero(mtx_drdi==-1,axis=0)

    row = pd.DataFrame({'pos':row_pos,'neg':row_neg})
    col = pd.DataFrame({'pos':col_pos,'neg':col_neg})

    row.to_csv('row.csv',index=False)
    col.to_csv('cal.csv',index=False)

    return neg


def split_k_fold(data_name, seed,split):

    train = {}
    valid = {}
    test = {}

    if os.path.exists(f'../../dataset/{data_name}/data_fold/{split}/fold_{0}_{seed}_train.csv'):
        for i in range(10):
            train[i] = pd.read_csv(f'../../dataset/{data_name}/data_fold/{split}/fold_{i}_{seed}_train.csv')
            valid[i] = pd.read_csv(f'../../dataset/{data_name}/data_fold/{split}/fold_{i}_{seed}_valid.csv')
            test[i] = pd.read_csv(f'../../dataset/{data_name}/data_fold/{split}/fold_{i}_{seed}_test.csv')
        return train, valid, test

    root_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(root_path)
    used_path = os.path.dirname(parent_path)

    data = pd.read_csv(f'../../dataset/{data_name}/drug_disease.csv')[['drug','indication']]
    num = len(data)

    drug_df = pd.read_csv(f'../../dataset/{data_name}/drug_dict.csv')
    dr = list(drug_df.iloc[:, 0])
    disease_df = pd.read_csv(f'../../dataset/{data_name}/disease_dict.csv')
    di = list(disease_df.iloc[:, 0])
    adjacency_matrix = pd.DataFrame(0, index=dr, columns=di)

    if os.path.exists(f'../../dataset/{data_name}/dda_matrix.txt'):
        matrix = np.loadtxt(f'../../dataset/{data_name}/dda_matrix.txt', delimiter='\t')
    else:
        for _, row in data.iterrows():
            drug = row['drug']
            disease = row['indication']
            adjacency_matrix.loc[drug, disease] = 1
            #print(_)

        # row = data['drug'].tolist()
        # col = data['indication'].tolist()
        # #print(positions)
        # #'DB01183', 'D014202'), ('DB00567', 'D008107')
        # adjacency_matrix.loc[row,col] = 1
        matrix = adjacency_matrix.values
        count_ones = np.sum(matrix == 1)
        #print('count_ones',count_ones)
        np.savetxt(f'../../dataset/{data_name}/dda_matrix.txt', matrix, delimiter='\t')


    drug_num, disease_num = matrix.shape[0], matrix.shape[1]
    drug_id, disease_id = np.nonzero(matrix)  # 矩阵非零元素索引,药物-疾病矩阵中=1的 药物序号和疾病序号

    num_len = int(np.ceil(len(drug_id) * 1))  # setting sparse ratio 药物疾病对个数
    drug_id, disease_id = drug_id[0: num_len], disease_id[0: num_len]  # 药物、疾病对序号
    #print(num_len)
    neutral_flag = 0
    labels = np.full((drug_num, disease_num), neutral_flag, dtype=np.int32)  # 药物×疾病大小的label矩阵
    observed_labels = [1] * len(drug_id)
    labels[drug_id, disease_id] = np.array(observed_labels)  # drug-disease有关联的=1, 相当于关联矩阵
    labels = labels.reshape([-1])  # 铺平

    pos_pairs = np.array([[dr, di] for dr, di in zip(drug_id, disease_id)])  # pos drug-disease (x,y) （drug,disease)
    pos_idx = np.array([dr * disease_num + di for dr, di in pos_pairs])  # 正样本位置记录数组

    #print('neg')
    # negative sampling
    neg_drug_idx, neg_disease_idx = np.where(matrix == 0)
    neg_pairs = np.array([[dr, di] for dr, di in zip(neg_drug_idx, neg_disease_idx)])  # neg drug-disease
    np.random.seed(6)
    np.random.shuffle(neg_pairs)

    if split =='warm':
        neg_pairs = neg_pairs[0:len(data)]
        neg_idx = np.array([dr * disease_num + di for dr, di in neg_pairs])
    else:
        neg_pairs = neg_pairs[0:len(data) * 10]

    data['label'] =[int(1)]*len(data)
    for i ,j in neg_pairs:
        data.loc[len(data.index)] = [ dr[i],di[j],0 ]

    #data.to_csv('../drug_disease_all.csv',index=False)
    #data.loc[4753:].to_csv('../drug_disease_neg.csv',index=False)

    if split=='mine':
        neg = generate_mine_neg(matrix, pos_pairs)
        df_neg = pd.DataFrame(columns=['drug','indication','label'])
        for u ,v in neg_pairs:
            df_neg.loc[len(df_neg.index)] = [dr[u],di[v],0]
        train, valid, test = data_split(data.loc[:num-1], df_neg, seed, 'warm', drug_df, disease_df,data_name)
    else:
        train,valid,test = data_split(data.loc[:num-1], data.loc[num:],seed,split,drug_df,disease_df,data_name)


    return train,valid,test




if __name__ == '__main__':
    split_data_dict = split_k_fold('MSI_new', 42,'warm')
