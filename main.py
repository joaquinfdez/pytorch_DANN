"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.optim as optim

import numpy as np

# from models import models
from train import test, train_model, params
from util import utils
from sklearn.manifold import TSNE

import argparse, sys

import torch
# from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from time import strftime



def visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                         tgt_test_dataloader, num_of_samples=None, imgName=None):
    """
    Evaluate the performance of dann and source only by visualization.

    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :param num_of_samples: the number of samples (from train and test respectively) for t-sne
    :param imgName: the name of saving image

    :return:
    """

    # Setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    # Randomly select samples from source domain and target domain.
    if num_of_samples is None:
        num_of_samples = params.batch_size
    else:
        assert len(src_test_dataloader) * num_of_samples, \
            'The number of samples can not bigger than dataset.' # NOT PRECISELY COMPUTATION

    # Collect source data.
    s_images, s_labels, s_tags = [], [], []
    for batch in src_test_dataloader:
        images, labels = batch

        if params.use_gpu:
            s_images.append(images.cuda())
        else:
            s_images.append(images)
        s_labels.append(labels)

        s_tags.append(torch.zeros((labels.size()[0])).type(torch.LongTensor))

        if len(s_images * params.batch_size) > num_of_samples:
            break

    s_images, s_labels, s_tags = torch.cat(s_images)[:num_of_samples], \
                                 torch.cat(s_labels)[:num_of_samples], torch.cat(s_tags)[:num_of_samples]


    # Collect test data.
    t_images, t_labels, t_tags = [], [], []
    for batch in tgt_test_dataloader:
        images, labels = batch

        if params.use_gpu:
            t_images.append(images.cuda())
        else:
            t_images.append(images)
        t_labels.append(labels)

        t_tags.append(torch.ones((labels.size()[0])).type(torch.LongTensor))

        if len(t_images * params.batch_size) > num_of_samples:
            break

    t_images, t_labels, t_tags = torch.cat(t_images)[:num_of_samples], \
                                 torch.cat(t_labels)[:num_of_samples], torch.cat(t_tags)[:num_of_samples]

    # Compute the embedding of target domain.
    embedding1 = feature_extractor(s_images)
    embedding2 = feature_extractor(t_images)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)

    if params.use_gpu:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.cpu().detach().numpy(),
                                                       embedding2.cpu().detach().numpy())))
    else:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.detach().numpy(),
                                                   embedding2.detach().numpy())))


    utils.plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)),
                         np.concatenate((s_tags, t_tags)), 'Domain Adaptation', imgName)


def display_writer(dict_train, dict_test, writer):


    # Tensorboard metrics
    # Record loss into the writer
    writer.add_scalars('Class_label_loss',
                        {'train': dict_train["class_label_loss"][-1],
                        'test': dict_test["class_label_loss"][-1],
                        }, dict_train["epoch"][-1])  
    writer.add_scalars('Domain_label_loss_src',
                        {'train': dict_train["domain_label_loss_src"][-1],
                        'test': dict_test["domain_label_loss_src"][-1],
                        }, dict_train["epoch"][-1])        
    writer.add_scalars('Domain_label_loss_tgt',
                        {'train': dict_train["domain_label_loss_tgt"][-1],
                        'test': dict_test["domain_label_loss_tgt"][-1],
                        }, dict_train["epoch"][-1])   
    writer.add_scalars('Domain_label_loss',
                        {'train': dict_train["domain_label_loss_tgt"][-1] + dict_train["domain_label_loss_src"][-1],
                        'test': dict_test["domain_label_loss_tgt"][-1] + dict_test["domain_label_loss_src"][-1],
                        }, dict_train["epoch"][-1])  
    writer.add_scalars('Metrics',
                       {
                        "source_correct": dict_test["source_correct"][-1], 
                        "target_correct": dict_test["target_correct"][-1], 
                        "domain_correct": dict_test["domain_correct"][-1] 
                       }, dict_train["epoch"][-1])  
    writer.add_scalars('Learning-Rate',
                          {'LR': params.lr,
                          'Momentum' :  params.momentum
                          }, dict_train["epoch"][-1])
    
    writer.flush()

    writer.close()

    print('Tensorboard is recording into folder.')
    
def load_writer(dict_train, dict_test, writer):


    for i in range(len(dict_train["class_label_loss"])):
        # Tensorboard metrics
        # Record loss into the writer
        writer.add_scalars('Class_label_loss',
                            {'train': dict_train["class_label_loss"][i],
                            'test': dict_test["class_label_loss"][i],
                            }, dict_train["epoch"][i])  
        writer.add_scalars('Domain_label_loss_src',
                            {'train': dict_train["domain_label_loss_src"][i],
                            'test': dict_test["domain_label_loss_src"][i],
                            }, dict_train["epoch"][i])        
        writer.add_scalars('Domain_label_loss_tgt',
                            {'train': dict_train["domain_label_loss_tgt"][i],
                            'test': dict_test["domain_label_loss_tgt"][i],
                            }, dict_train["epoch"][i])   
        writer.add_scalars('Domain_label_loss',
                            {'train': dict_train["domain_label_loss_tgt"][i] + dict_train["domain_label_loss_src"][i],
                            'test': dict_test["domain_label_loss_tgt"][i] + dict_test["domain_label_loss_src"][i],
                            }, dict_train["epoch"][i])  
        writer.add_scalars('Metrics',
                           {
                            "source_correct": dict_test["source_correct"][i], 
                            "target_correct": dict_test["target_correct"][i], 
                            "domain_correct": dict_test["domain_correct"][i] 
                           }, dict_train["epoch"][i])  
        writer.add_scalars('Learning-Rate',
                              {'LR': params.lr,
                              'Momentum' :  params.momentum
                              }, dict_train["epoch"][i])
        
        writer.flush()

    writer.close()

    print('Tensorboard is recording into folder.')

    
def main(args):

    # Set global parameters.
    params.fig_mode = args.fig_mode
    params.epochs = args.max_epoch
    params.training_mode = args.training_mode
    params.source_domain = args.source_domain
    params.target_domain = args.target_domain
    if params.embed_plot_epoch is None:
        params.embed_plot_epoch = args.embed_plot_epoch
    params.lr_initial = args.lr
    params.lr = params.lr_initial
    params.momentum = args.momentum
    params.neural_network_name = args.neural_network_name
    params.load = args.load
    params.epoch_init = args.epoch_init if utils.string_to_boolean(params.load) else 0

    flag = True
    """
        Tensorboard
    """
    PATH_to_log_dir = './runs/' + params.neural_network_name + '/'
    utils.mkdirs(PATH_to_log_dir)
    timestr = strftime("%m%d_%H%M")
    writer = SummaryWriter(PATH_to_log_dir + params.neural_network_name + '(' + timestr + ')')
    print('Tensorboard is recording into folder: ' + PATH_to_log_dir + params.neural_network_name + timestr)

    if args.save_dir is not None:
        params.save_dir = args.save_dir
    else:
        print('Figures will be saved in ./experiment folder.')

    # prepare the source data and target data

    src_train_dataloader = utils.get_train_loader(params.source_domain)
    src_test_dataloader = utils.get_test_loader(params.source_domain)
    tgt_train_dataloader = utils.get_train_loader(params.target_domain)
    tgt_test_dataloader = utils.get_test_loader(params.target_domain)

    if params.fig_mode is not None:
        print('Images from training on source domain:')

        utils.displayImages(src_train_dataloader, imgName='source')

        print('Images from test on target domain:')
        utils.displayImages(tgt_test_dataloader, imgName='target')

    # init metrics
    dict_train, dict_test = utils.get_training_info(utils.string_to_boolean(params.load))
    
    # init models
    model_index = params.source_domain + '_' + params.target_domain
    feature_extractor = params.extractor_dict[model_index]
    class_classifier = params.class_dict[model_index]
    domain_classifier = params.domain_dict[model_index]

    if params.use_gpu:
        feature_extractor.cuda()
        class_classifier.cuda()
        domain_classifier.cuda()

    # init criterions
    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()

    # init optimizer
    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()},
                            {'params': domain_classifier.parameters()}], lr= params.lr, momentum= params.momentum)

    # Loading previous 
    if utils.string_to_boolean(params.load):
        print("Loading weights in models.")
        _models = feature_extractor, class_classifier, domain_classifier
        _models = utils.load_pytorch_models(_models, params.epoch_init)
        feature_extractor, class_classifier, domain_classifier = _models
        
        load_writer(dict_train, dict_test, writer)
        
    # Training process
    for epoch in range(params.epoch_init, params.epochs):
        print('Epoch: {}'.format(epoch))
        dict_train, _models = train_model.train(args.training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
                             src_train_dataloader, tgt_train_dataloader, optimizer, epoch, writer, dict_train, flag)
        dict_test = test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader, class_criterion, domain_criterion, writer, epoch, dict_test)

        display_writer(dict_train, 
                       dict_test, 
                       writer)
        # Save models periodically
        if epoch % 1 == 0:
            # if epoch != epoch_init:  
            utils.save_pytorch_models(_models, epoch)            
            # Saving dictionaries
            utils.save_training_info(dict_train, dict_test)
        # Plot embeddings periodically.
        if epoch % params.embed_plot_epoch == 0 and params.fig_mode is not None:
            visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                                 tgt_test_dataloader, imgName='embedding_' + str(epoch))



def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', type= str, default= 'MNIST', help= 'Choose source domain.')

    parser.add_argument('--target_domain', type= str, default= 'MNIST_M', help = 'Choose target domain.')

    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')

    parser.add_argument('--save_dir', type=str, default=None, help='Path to save plotted images.')

    parser.add_argument('--training_mode', type=str, default='dann', help='Choose a mode to train the model.')

    parser.add_argument('--max_epoch', type=int, default=1000, help='The max number of epochs.')

    parser.add_argument('--embed_plot_epoch', type= int, default=100, help= 'Epoch number of plotting embeddings.')

    parser.add_argument('--lr', type= float, default= 0.01, help= 'Learning rate.')

    parser.add_argument('--momentum', type= float, default= 0.9, help= 'Momentum.')

    parser.add_argument('--neural_network_name', type=str, default='dann', help='Choose a neural network name.')

    parser.add_argument('--load', type=str, default='True', help='Select train or retrain (False or True)')

    parser.add_argument('--epoch_init', type=int, default=15, help='Init')

    

    return parser.parse_args()



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
