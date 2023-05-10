from typing import Any
import numpy as np
import torch
from torch._C import device
import torch.optim as optim
from multiprocessing import Pool
import dataloader
# import model.dmt_model as model_small
import dmt_model_big as model_big
import os
import paramENC as paramzzl
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tool
import scipy.sparse as sp
import logging
from torch.nn import functional as F
import pandas
import matplotlib.pyplot as plt
import time


def train(
    args : dict,
    Model : Any, 
    data: Any,
    target: Any,
    optimizer: Any,
    epoch: int):

    BATCH_SIZE = args['batch_size']

    Model.train()

    num_train_sample = data.shape[0]
    num_batch = (num_train_sample - 0.5) // BATCH_SIZE + 1

    rand_index_i = torch.randperm(num_train_sample, device=Model.device).long()
    train_loss_sum = [0, 0, 0, 0, 0, 0, 0]

    for batch_idx in torch.arange(0, num_batch):
        start = (batch_idx * BATCH_SIZE).int().to(Model.device)
        end = torch.min(
            torch.tensor(
                [batch_idx * BATCH_SIZE + BATCH_SIZE,
                 num_train_sample])).to(Model.device)
        sample_index_i = rand_index_i[start:end.int()]

        optimizer.zero_grad()
        input_data = data[sample_index_i].to(Model.device)
        output = Model(input_data, sample_index_i)
        loss_list = Model.Loss(output, sample_index_i, data=input_data)

        loss_list[0].backward()
        loss_list[1].backward()
        train_loss_sum[0] += loss_list[0].item()
        train_loss_sum[1] += loss_list[1].item()

        if (args['trainquiet'] == 0) and (epoch % 10 == 0):
            print('batch {} loss {}'.format(batch_idx, loss_list[0].item()))

        optimizer.step()

    if args['trainquiet'] == 0 and (epoch % 100 == 0):
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {}'.format(
            epoch, batch_idx * BATCH_SIZE, num_train_sample,
            BATCH_SIZE * 100. * batch_idx / num_train_sample, train_loss_sum))
        print(Model.vList[-1])

    return train_loss_sum


def Test(
    args : dict,
    Model : Any, 
    data: Any,
    target: Any,
    optimizer: Any,
    epoch: int):

    Model.eval()
    BATCH_SIZE = args['batch_size']
    num_train_sample = data.shape[0]
    num_batch = (num_train_sample - 0.5) // BATCH_SIZE + 1
    rand_index_i = torch.arange(num_train_sample)

    for batch_idx in torch.arange(0, num_batch):
        start = (batch_idx * BATCH_SIZE).int()
        end = torch.min(
            torch.tensor(
                [batch_idx * BATCH_SIZE + BATCH_SIZE, num_train_sample]))
        sample_index_i = rand_index_i[start:end.int()]

        datab = data.float()[sample_index_i]
        em = Model.test(datab)
        re = Model.Generate(em[-1])
        # print(em[0])

        em = em[-1].detach().cpu().numpy()
        re = re[-1].detach().cpu().numpy()
        if batch_idx == 0:
            outem = em
            outre = re
        else:
            outem = np.concatenate((outem, em), axis=0)
            outre = np.concatenate((outre, re), axis=0)

        # Model.Loss(em, rand_index_i)

    return outem, outre

def masked_loss(out, label, mask):
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss

def normalize_S(S):
    """Symmetrically normalize similarity matrix."""
    S = sp.coo_matrix(S)
    rowsum = np.array(S.sum(1)) # D
    for i in range(len(rowsum)):
        if rowsum[i] == 0:
            print("error")
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return S.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5SD^0.5


def noise_train(args, data, label_one_hot, Model, adj, support, y_label, optimizer,train_mask, test_mask,
                              epoch, best_acc):
    device = args['device']
    BATCH_SIZE = args['batch_size']

    Model.train()

    num_train_sample = torch.sum(train_mask)
    num_batch = (num_train_sample - 0.5) // BATCH_SIZE + 1

    optimizer.zero_grad()

    latent = Model(data)

    v = torch.mean((latent[adj.row] - latent[adj.col]) ** 2, 1)
    eps = np.finfo(float).eps
    v = 1 / (v + eps)
    v = v.detach().cpu().numpy()
    S = sp.coo_matrix((v, (adj.row, adj.col)), shape=adj.shape)
    S = normalize_S(S)
    label_sp = sp.lil_matrix(label_one_hot.detach().cpu().numpy())
    for j in range(50):
        if j == 0:
            Z = S.dot(label_sp)
        else:
            Z = S.dot(Z)
    Z = torch.from_numpy(Z.toarray()).to(device)

    Z = F.softmax(Z, dim=1)
    Z_train = Z.argmax(dim=1)[train_mask]
    tmp_mask = Z_train == y_label[train_mask]
    # a = torch.sum(tmp_mask)

    pseudo_mask = torch.zeros(latent.shape[0]).bool().to(device)
    pseudo_mask[:num_train_sample] = tmp_mask
    ones = torch.eye(latent.shape[1]).to(device)
    pseudo_y_onehot = torch.zeros_like(label_one_hot).to(device)
    pseudo_y_onehot[pseudo_mask] = ones.index_select(0, y_label[pseudo_mask])

    loss, y_hat_ = Model.Loss(latent, data, pseudo_y_onehot, train_mask, support, num_train_sample)
    # train_loss = loss[0] + args["mu_V"]*loss[1] + args["mu_A"]*loss[2]
    train_loss = loss[1]

    train_loss.backward()
    optimizer.step()
    train_acc = tool.acc(y_hat_, y_label, train_mask)

    Model.eval()
    # test_mask = ~train_mask
    num_test_sample = torch.sum(test_mask)
    latent = Model(data)
    loss_, y_hat_ = Model.Loss(latent, data, pseudo_y_onehot, test_mask, support, num_test_sample)
    # test_loss = loss_[0] + args["mu_V"]*loss_[1] + args["mu_A"]*loss_[2]
    test_loss = loss_[1]

    test_acc = tool.acc(y_hat_, y_label, test_mask)

    if (args['trainquiet'] == 0) and (epoch % 1000 == 0):
        logging.info('Epoch: {:04d}, train_loss:{:.4f}, Loss_DMT:{:.4f}, Loss_V:{:.4f}, Loss_A:{:.4f}, train_acc: {:.4f},'
                     'test_loss: {:.4f}, Loss_DMT_test:{:.4f}, Loss_V_test:{:.4f}, Loss_A_test:{:.4f}, test_acc: {:.4f}'
                     .format(epoch, train_loss.item(), loss[0].item(), loss[1].item(), loss[2].item(), train_acc,
                      test_loss.item(), loss_[0].item(), loss_[1].item(), loss_[2].item(), test_acc))
        logging.info('v:{}'.format(Model.vList[-1]))


    if test_acc >= best_acc:
        best_acc = test_acc

    return train_loss, best_acc


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def count_paras(model, logging=None):
    '''
    Count model parameters.
    '''
    nparas = sum(p.numel() for p in model.parameters())
    if logging is None:
        print ('#paras of my model: total {}M'.format(nparas/1e6))
    else:
        logging.info('#paras of my model: total {}M'.format(nparas/1e6))
    return nparas


def main(args, seed, prob, log=True):
    # Use only the specified GPU
    n_gpu = len(args['gpu_id'].split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']
    # device = torch.device('cuda:{}'.format(args['gpu_id']) if torch.cuda.is_available() else 'cpu')

    path = tool.GetPath(args['data_name'] + '_' + args['name'])
    tool.SetSeed(seed)

    if args['isTeacher']:
        if args['teacher_path'] is None:
            exp_name = path + '{}_{}{:.1f}_seed{}_best.pth'.format(args['data_name'], args['corruption_type'],
                                                                             prob, seed)
        else:
            args['teacher_path'] = path + '{}_{}{:.1f}_seed{}_best.pth'.format(args['data_name'], args['corruption_type'],
                                                                   prob, seed)
            exp_name = path + '{}_{}{:.1f}_seed{}_student'.format(args['data_name'], args['corruption_type'],
                                                                             prob, seed)
    else:
        exp_name = path + '{}_{}{:.1f}_seed{}'.format(args['data_name'], args['corruption_type'],
                                                                         prob, seed)
    ## set up the logger
    if log:
        tool.set_logger(os.path.join(path, 'train.log'))

    adj, features, y_clean_onehot, mask = dataloader.GetData(args)

    adj = adj + sp.eye(adj.shape[0])
    # adj = sp.coo_matrix(adj)
    if args['way'] in ['lpa', 'GCN']:
        support = sp.coo_matrix(adj)

        # support = torch.FloatTensor(adj.todense()).float().cuda()
    else:
        support = dataloader.preprocess_adj(adj)
        support = torch.FloatTensor(support.todense()).float().cuda()
    num_classes = y_clean_onehot.shape[1]
    num_nodes = y_clean_onehot.shape[0]
    y_true = y_clean_onehot.argmax(dim=1).cuda()
    y_label = torch.max(y_clean_onehot, dim=1)[1].cuda()

    if args["corruption_type"] == 'uniform':
        C = tool.uniform_mix_C(prob, num_classes)
    elif args["corruption_type"] == 'flip':
        C = tool.flip_labels_C(prob, num_classes)

    for i in mask['indices']:
        y_label[i] = np.random.choice(num_classes, p=C[y_label[i]])

    if args['way'] == 'teaching':
        NoiseTeaching = model_big.Noise_Teaching(features, args, path,
                                           y_label, y_true, num_nodes, num_classes,
                                           adj, support, **mask)
    else:
        test_accs = []
        best_acc = 0.
        NoiseTrain = model_big.Noise_Train(features, args, path,
                                           y_label, num_nodes, num_classes,
                                           adj, support, **mask)
        if n_gpu > 1:
            NoiseTrain = torch.nn.DataParallel(NoiseTrain)
        NoiseTrain.cuda()
        optimizer = optim.Adam(NoiseTrain.parameters(), lr=0.01, weight_decay=0.0005)
        nparas = count_paras(NoiseTrain, logging)

    if args['way'] == 'DMT':

        if args['teacher_path'] is not None:
            NoiseTrain.load_state_dict(torch.load(args['teacher_path'], map_location='cuda:0'))
            NoiseTrain.eval()
            out = NoiseTrain()
            out_ = F.softmax(out, dim=1)
            NoiseTrain.y_teacher = out_
            New_Z = NoiseTrain.train_with_noise(out)
            args['epochs'] = 200
            # if args['isWeight']:
            import scipy.stats
            # out_ = out_.detach().cpu().numpy()
            # weights = scipy.stats.entropy(out_.T)
            # weights = 1 - weights / np.log(out_.shape[1])
            # weights = weights / np.max(weights)
            # weights = torch.tensor(weights).cuda()
            ###
            # w2 = weights.unsqueeze(dim=-1)
            # w2 = w2.repeat(1, num_classes)
            # w2_ = NoiseTrain.weights
            # w2_ = w2_.unsqueeze(dim=-1).repeat(1, num_classes)
            # New_Z = New_Z * (w2_ - w2) + NoiseTrain.y_teacher * w2
            # New_Z = New_Z * (1- beta) + out_ * beta

            # New_Z = New_Z.argmax(dim=1)
            # NoiseTrain.isWeight = False
            ###
            # NoiseTrain.weights = weights
        else:
            New_Z = y_label

        for epoch in range(0, args['epochs'] + 1):
            NoiseTrain.train()
            optimizer.zero_grad()
            out = NoiseTrain()
            train_loss = NoiseTrain.masked_loss(out, New_Z, mask['train_mask'])
            train_acc = tool.acc(out, y_label, mask['train_mask'])

            train_loss.backward()
            optimizer.step()

            NoiseTrain.eval()
            out = NoiseTrain()
            val_acc = tool.acc(out, y_label, mask['val_mask'])
            test_acc = tool.acc(out, y_label, mask['test_mask'])
            if epoch % 100 == 0:
                if log:
                    logging.info(
                        'Epoch: {:04d}, train_loss:{:.4f}, train_acc: {:.4f}, test_acc: {:.4f}'
                            .format(epoch, train_loss.item(), train_acc,test_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                acc_best_test = test_acc
                best_epoch = epoch
                if args['save_model']:
                    torch.save(NoiseTrain.state_dict(), '{}'.format(exp_name))

            NoiseTrain.epoch = epoch
        if log:
            logging.info('{}_{}{:.1f}_seed{}'.format(args['data_name'], args['corruption_type'],
                                                                         prob, seed))
            logging.info('best accuracy:{} is achieved at {} epoch'.format(best_acc, best_epoch))
        return acc_best_test

    elif args['way'] == 'lpa':
        for epoch in range(0, args['epochs'] + 1):
            NoiseTrain.train()
            optimizer.zero_grad()
            out = NoiseTrain()
            train_loss = NoiseTrain.masked_loss(out, y_label, mask['train_mask'])
            train_acc = tool.acc(out, y_label, mask['train_mask'])

            train_loss.backward(retain_graph=True)
            optimizer.step()

            NoiseTrain.eval()
            out = NoiseTrain()
            val_acc = tool.acc(out, y_label, mask['val_mask'])
            if epoch % 100 == 0:
                if log:
                    logging.info(
                        'Epoch: {:04d}, train_loss:{:.4f}, train_acc: {:.4f}, val_acc: {:.4f}'
                            .format(epoch, train_loss.item(), train_acc, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                NoiseTrain.eval()
                out = NoiseTrain()
                test_acc = tool.acc(out, y_label, mask['test_mask'])
                acc_best_test = test_acc
                best_epoch = epoch
                if args['save_model']:
                    torch.save(NoiseTrain.state_dict(), '{}'.format(exp_name))

            NoiseTrain.epoch = epoch
        if log:
            logging.info('{}_{}{:.1f}_seed{}'.format(args['data_name'], args['corruption_type'],
                                                                         prob, seed))
            logging.info('best accuracy:{} is achieved at {} epoch'.format(best_acc, best_epoch))
        return acc_best_test
    elif args['way'] == 'cluster':
        with torch.autograd.set_detect_anomaly(True):
            gifPloterLatentTrain = tool.GIFPloter()
            loss_his = []
            for epoch in range(0, args['epochs'] + 1):
                NoiseTrain.train()
                optimizer.zero_grad()
                out = NoiseTrain()
                train_loss = NoiseTrain.Loss(out, mask['train_mask'])
                train_acc = tool.acc(out, y_label, mask['train_mask'])

                train_loss.backward()
                optimizer.step()
                loss_his.append(train_loss.item())

                NoiseTrain.eval()
                out = NoiseTrain()

                NoiseTrain.epoch = epoch
                test_acc = tool.acc(out, y_label, mask['test_mask'])

                if epoch % 100 == 0:
                    if log:
                        logging.info(
                            'Epoch: {:04d}, train_loss:{:.4f}, train_acc: {:.4f}, test_acc: {:.4f}'
                                .format(epoch, train_loss.item(), train_acc,test_acc))
                    gifPloterLatentTrain.AddNewFig(
                        out.detach().cpu().numpy(),
                        y_true.detach().cpu(),
                        his_loss=loss_his,
                        path=path,
                        graph=NoiseTrain.GetInput(),
                        link=None,
                        title_='train_epoch_em{}_{}{}.png'.format(
                            epoch, args['perplexity'], args['v']))
                    # tool.SaveData(
                    #     features,
                    #     out,
                    #     y_label,
                    #     path=path,
                    #     name='train_epoch{}'.format(str(epoch).zfill(6)))

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                    if args['save_model']:
                        torch.save(NoiseTrain.state_dict(), '{}'.format(exp_name))
                print('epoch:', epoch)
                NoiseTrain.epoch = epoch
                if log:
                    logging.info('{}_{}{:.1f}_seed{}'.format(args['data_name'], args['corruption_type'],
                                                             prob, seed))
                    logging.info('best accuracy:{} is achieved at {} epoch'.format(best_acc, best_epoch))

            gifPloterLatentTrain.SaveGIF()

    elif args['way'] == 'teaching':
        acc_list = []
        best_acc, epoch_best = 0., 0.
        for epoch in range(1, args['epochs']):
            NoiseTeaching.epoch = epoch
            NoiseTeaching.model1.epoch = epoch
            NoiseTeaching.model2.epoch = epoch
            # train models
            train_acc1, train_acc2, pure_ratio = NoiseTeaching.train()

            # evaluate models
            test_acc1, test_acc2 = NoiseTeaching.evaluate()

            if epoch >= 290:
                acc_list.extend([test_acc1, test_acc2])

            if max(test_acc1, test_acc2) > best_acc:
                index = np.argmax([test_acc1, test_acc2])
                best_acc = max(test_acc1, test_acc2)
                epoch_best = epoch
                if args['save_model']:
                    torch.save([NoiseTeaching.model1.state_dict(), NoiseTeaching.model2.state_dict()][index], '{}_best.pth'
                               .format(exp_name))
            if epoch % 100 == 0:
                # save results
                logging.info('Epoch [%d/%d] test acc: Model1 %.4f %% Model2 %.4f %%, Pure Ratio%.4f %%,'
                    'best_acc: %.4f, Epoch_best: %d' % (
                        epoch + 1, args['epochs'], test_acc1, test_acc2, pure_ratio, best_acc, epoch_best))

        avg_acc = sum(acc_list) / len(acc_list)
        logging.info("the average acc in last 10 epochs: {}".format(str(avg_acc)))
        logging.info('best accuracy:{} is achieved at {} epoch'.format(best_acc, epoch_best))

    elif args['way'] == 'GCN':
        for epoch in range(0, args['epochs'] + 1):
            NoiseTrain.train()
            optimizer.zero_grad()
            out = NoiseTrain()
            train_loss = NoiseTrain.masked_loss(out, y_label, mask['train_mask'])
            train_acc = tool.acc(out, y_label, mask['train_mask'])

            train_loss.backward(retain_graph=True)
            optimizer.step()

            NoiseTrain.eval()
            out = NoiseTrain()
            val_acc = tool.acc(out, y_label, mask['val_mask'])
            test_acc = tool.acc(out, y_label, mask['test_mask'])
            if epoch % 100 == 0:
                if log:
                    logging.info(
                        'Epoch: {:04d}, train_loss:{:.4f}, train_acc: {:.4f}, test_acc: {:.4f}'
                            .format(epoch, train_loss.item(), train_acc, test_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                acc_best_test = test_acc
                best_epoch = epoch
                if args['save_model']:
                    torch.save(NoiseTrain.state_dict(), '{}'.format(exp_name))

            NoiseTrain.epoch = epoch
        if log:
            logging.info('{}_{}{:.1f}_seed{}'.format(args['data_name'], args['corruption_type'],
                                                     prob, seed))
            logging.info('best accuracy:{} is achieved at {} epoch'.format(best_acc, best_epoch))
        return acc_best_test
    else:
        raise ValueError

def main_label_agg_weight():
    args = paramzzl.GetParamCora()
    # args = paramzzl.GetParamCiteseer()
    # args = paramzzl.GetParamPubmed()
    # args = paramzzl.GetParamCoauthorPhy()

    name = '{}_{}'.format(args['data_name'], args['corruption_type'])
    result_dict = {}
    result_dict['name'] = name
    for prob in args['corruption_prob']:
        accs = []
        for seed in args['seed']:
            acc = main(args, seed, prob)
            accs.append(acc)
        mean = np.mean(accs, dtype=float)
        std = np.std(accs, dtype=float)
        mean_std = '%.2f(%.2f)' % (mean * 100, std * 100)
        result_dict[prob] = mean_std
    logging.info(result_dict)

main_label_agg_weight()



