import torch
from torch.autograd import Variable
import numpy as np

from train import params
from util import utils

import torch.optim as optim


def train(training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
          source_dataloader, target_dataloader, optimizer, epoch, writer, dict_train, flag):
    """
    Execute target domain adaptation
    :param training_mode:
    :param feature_extractor:
    :param class_classifier:
    :param domain_classifier:
    :param class_criterion:
    :param domain_criterion:
    :param source_dataloader:
    :param target_dataloader:
    :param optimizer:
    :return:
    """    
    # setup models
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    # steps
    start_steps = epoch * len(source_dataloader)
    total_steps = params.epochs * len(source_dataloader)

    # Setup metrics
    class_label_loss = 0.0 # Class label  
    domain_label_loss_src = 0.0 # Domain real 
    domain_label_loss_tgt = 0.0 # Domain that we want to adapt 

    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):

        if training_mode == 'dann':
            # setup hyperparameters
            p = float(batch_idx + start_steps) / total_steps
            print("p-->", p)
            constant = 2. / (1. + np.exp(-params.gamma * p)) - 1
            print("constant-->", constant)

            # prepare the data
            input1, label1 = sdata
            input2, label2 = tdata
            size = min((input1.shape[0], input2.shape[0]))
            input1, label1 = input1[0:size, :, :, :], label1[0:size]
            input2, label2 = input2[0:size, :, :, :], label2[0:size]
            if params.use_gpu:
                input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
                input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            else:
                input1, label1 = Variable(input1), Variable(label1)
                input2, label2 = Variable(input2), Variable(label2)

            # setup optimizer
            # optimizer = utils.optimizer_scheduler(optimizer, p)
            optimizer.zero_grad()

            # prepare domain labels
            if params.use_gpu:
                source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())
                target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
            else:
                source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))
                target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

            # compute the output of source domain and target domain
            src_feature = feature_extractor(input1)
            tgt_feature = feature_extractor(input2)

            # compute the class loss of src_feature
            class_preds = class_classifier(src_feature)
            class_loss = class_criterion(class_preds, label1)
            class_label_loss += class_loss.item()

            # compute the domain loss of src_feature and target_feature
            tgt_preds = domain_classifier(tgt_feature, constant)
            src_preds = domain_classifier(src_feature, constant)
            tgt_loss = domain_criterion(tgt_preds, target_labels)
            src_loss = domain_criterion(src_preds, source_labels)
            domain_loss = tgt_loss + src_loss

            domain_label_loss_src += src_loss.item()
            domain_label_loss_tgt += tgt_loss.item()

            loss = class_loss + params.theta * domain_loss
            loss.backward()
            optimizer.step()
            
            # Saving graph model
            # if(flag):
            #     # Writer of model
            #     writer.add_graph(feature_extractor, (input1))
            #     writer.flush()

            #     writer.add_graph(class_classifier, (src_feature))
            #     writer.flush()

            #     writer.add_graph(domain_classifier, (src_feature, constant))
            #     writer.flush()
            #     flag = False

            # print loss
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(target_dataloader.dataset),
                    100. * batch_idx / len(target_dataloader), loss.item(), class_loss.item(),
                    domain_loss.item()
                ))


        elif training_mode == 'source':
            # prepare the data
            input1, label1 = sdata
            size = input1.shape[0]
            input1, label1 = input1[0:size, :, :, :], label1[0:size]

            if params.use_gpu:
                input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            else:
                input1, label1 = Variable(input1), Variable(label1)

            # setup optimizer
            optimizer = optim.SGD(list(feature_extractor.parameters())+list(class_classifier.parameters()), lr=0.01, momentum=0.9)
            optimizer.zero_grad()

            # compute the output of source domain and target domain
            src_feature = feature_extractor(input1)

            # compute the class loss of src_feature
            class_preds = class_classifier(src_feature)
            class_loss = class_criterion(class_preds, label1)
            class_label_loss += class_loss.item()

            class_loss.backward()
            optimizer.step()

            # print loss
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input1), len(source_dataloader.dataset),
                    100. * batch_idx / len(source_dataloader), class_loss.item()
                ))

        elif training_mode == 'target':
            # prepare the data
            input2, label2 = tdata
            size = input2.shape[0]
            input2, label2 = input2[0:size, :, :, :], label2[0:size]
            if params.use_gpu:
                input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            else:
                input2, label2 = Variable(input2), Variable(label2)

            # setup optimizer
            optimizer = optim.SGD(list(feature_extractor.parameters()) + list(class_classifier.parameters()), lr=0.01,
                                  momentum=0.9)
            optimizer.zero_grad()

            # compute the output of source domain and target domain
            tgt_feature = feature_extractor(input2)

            # compute the class loss of src_feature
            class_preds = class_classifier(tgt_feature)
            class_loss = class_criterion(class_preds, label2)
            class_label_loss += class_loss.item()
            
            class_loss.backward()
            optimizer.step()

            # print loss
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(target_dataloader.dataset),
                    100. * batch_idx / len(target_dataloader), class_loss.item()
                ))

        class_label_loss /= len(source_dataloader)
        domain_label_loss_src /= len(source_dataloader)
        domain_label_loss_tgt /= len(source_dataloader)

    dict_train["class_label_loss"].append(class_label_loss)
    dict_train["domain_label_loss_src"].append(domain_label_loss_src)
    dict_train["domain_label_loss_tgt"].append(domain_label_loss_tgt)
    dict_train["epoch"].append(epoch)
    
    models = feature_extractor, class_classifier, domain_classifier
    return dict_train, models