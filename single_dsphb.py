#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 20:28:45 2022

https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/quickstart_tutorial/quickstart_tutorial.ipynb
"""
from __future__ import print_function
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import tweaked_torchaudio as MSHB
from functions import data_processing, GreedyDecoder, cer, wer, data_processing_train, data_processing_test
from single_net import DeepSpeechNNET
import torch.nn.functional as F
import utils
from torch.optim.lr_scheduler import StepLR

os.environ["PT_HPU_LAZY_MODE"] = "1"
from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()

# Download training data from open datasets.
# training_data = MSHB.LIBRISPEECH(root="data",train=True,download=True)

learning_rate=0.1
## Optimizers :
loss_fn   = nn.CTCLoss(blank = 28)   

def train(args,  model, device, train_dataloader, optimizer, epoch):
    size = len(train_dataloader.dataset)
    model.train()
    for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(train_dataloader,0):
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        spectrograms = spectrograms.contiguous(memory_format=torch.channels_last)
        if args.use_lazy_mode:
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()        
        # Compute prediction error
        output = model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        loss = loss_fn(output, labels, input_lengths, label_lengths)
        tensorboard_logs = {"Loss/train": loss}
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if args.use_lazy_mode:
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()
        
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            # loss, current = loss.item(), batch * len(input_lengths)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), len(train_dataloader.dataset)/args.world_size,
                100. * batch_idx / len(train_dataloader), loss.item()))            
            
            if args.dry_run:
                break

    #     tensorboard_logs = {"Loss/train": loss}
    return {"loss": loss, "log": tensorboard_logs}


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    n_correct_pred=0
    metric_logger = utils.MetricLogger(delimiter="  ",device=device)
    with torch.no_grad():
        for spectrograms, labels, input_lengths, label_lengths in test_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            pred = model(spectrograms)
            pred = F.log_softmax(pred, dim=2)
            pred = pred.transpose(0, 1)  # (time, batch, n_class)
            test_loss += loss_fn(pred, labels, input_lengths, label_lengths).item()
            decoded_preds, decoded_targets = GreedyDecoder(pred.transpose(0, 1), labels, label_lengths)

            batch_size = spectrograms.shape[0]
            # acc, _ = utils.accuracy(decoded_preds, decoded_targets, topk=(1, 5))
            acc = sum([int(a == b) for a, b in zip(decoded_preds, decoded_targets)])
            metric_logger.meters['acc'].update(acc, n=batch_size)
            
            test_cer, test_wer = [], []
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
    avg_wer = torch.FloatTensor([sum(test_wer) / len(test_wer)])  # Need workt to make all operations in torch
    logs = {"cer": avg_cer,"wer": avg_wer,}

    test_loss /= (len(test_loader.dataset)/args.world_size)
    metric_logger.meters['loss'].update(test_loss)
    metric_logger.synchronize_between_processes()
            
    print('\nTotal test set: {}, number of workers: {}'.format(len(test_loader.dataset), args.world_size))
    print('* Average Acc {top.global_avg:.3f} Average loss {value.global_avg:.3f}'.format(top=metric_logger.acc, value=metric_logger.loss))

    return {"val_loss": test_loss,"n_correct_pred": n_correct_pred, "n_pred": len(spectrograms),"log": logs,"wer": avg_wer,"cer": avg_cer,        }

# def validation(dataloa4der, model, loss_fn):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (spectrograms, labels, input_lengths, label_lengths) in enumerate(dataloader):
#         spectrograms, labels = spectrograms.to(device), labels.to(device)

#         y_hat = model(spectrograms)  # (batch, time, n_class)
#         output = F.log_softmax(y_hat, dim=2)
#         output = output.transpose(0, 1)  # (time, batch, n_class)
#         loss = loss_fn(output, labels, input_lengths, label_lengths)

#         decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
#         n_correct_pred = sum([int(a == b) for a, b in zip(decoded_preds, decoded_targets)])

#         test_cer, test_wer = [], []
#         for j in range(len(decoded_preds)):
#             test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
#             test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

#     avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
#     avg_wer = torch.FloatTensor([sum(test_wer) / len(test_wer)])  # Need workt to make all operations in torch
#     logs = {"cer": avg_cer,"wer": avg_wer,}
#     # self.log("wer", avg_wer)
#     return {"val_loss": loss,"n_correct_pred": n_correct_pred,"n_pred": len(spectrograms),"log": logs,"wer": avg_wer,"cer": avg_cer,}


def permute_params(model, to_filters_last, lazy_mode):
    import habana_frameworks.torch.core as htcore
    if htcore.is_enabled_weight_permute_pass() is True:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                if to_filters_last:
                    param.data = param.data.permute((2, 3, 1, 0))
                else:
                    param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS
    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()

def permute_momentum(optimizer, to_filters_last, lazy_mode):
    # Permute the momentum buffer before using for checkpoint
    import habana_frameworks.torch.core as htcore
    if htcore.is_enabled_weight_permute_pass() is True:
        return
    for group in optimizer.param_groups:
        for p in group['params']:
            param_state = optimizer.state[p]
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                if(buf.ndim == 4):
                    if to_filters_last:
                        buf = buf.permute((2,3,1,0))
                    else:
                        buf = buf.permute((3,2,0,1))
                    param_state['momentum_buffer'] = buf

    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DeepSpeech Toy Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,    help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=True,     help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=17, metavar='S',        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--hpu', action='store_true', default=True,         help='Use hpu device')
    parser.add_argument('--use_lazy_mode', action='store_true', default=True,help='Enable lazy mode on hpu device, default eager mode')
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='ops_bf16_mnist.txt', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='ops_fp32_mnist.txt', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    parser.add_argument('--dl-worker-type', default='MP', type=lambda x: x.upper(),choices = ["MT", "MP"], help='select multithreading or multiprocessing')
    parser.add_argument('--world_size', default=8, type=int, metavar='N',
                        help='number of total workers (default: 1)')
    parser.add_argument('--process-per-node', default=8, type=int, metavar='N',
                        help='Number of process per node')

    parser.add_argument('--distributed', action='store_true', help='whether to enable distributed mode and run on multiple devices')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(use_cuda)
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.multiprocessing.set_start_method('spawn')
    if args.hpu:
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        load_habana_module()
        device = torch.device("hpu")
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device     = lambda x: None

    if args.use_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"]="1"
        import habana_frameworks.torch.core as htcore

    if args.is_hmp:
        from habana_frameworks.torch.hpex import hmp
        hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                    fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)

    utils.init_distributed_mode(args)

    train_kwargs = {'batch_size' : args.batch_size}
    test_kwargs  = {'batch_size' : args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    dataset1       = MSHB.LIBRISPEECH("data_root", url="dev-clean", download=True)

    # validation_dataset       = MSHB.LIBRISPEECH("data_root", url="dev-clean", download=True)
    # validation_DataLoader = DataLoader(
    #             dataset      =validation_dataset,
    #             batch_size   =64,
    #             shuffle      =False,
    #             collate_fn   =lambda x: data_processing(x, "valid"),
    #             num_workers  =10,
    #         )

    dataset2      = MSHB.LIBRISPEECH("data_root", url="dev-clean", download=True)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset1)
        test_sampler  = torch.utils.data.distributed.DistributedSampler(dataset2)
        
        train_loader = DataLoader(dataset      =dataset1,
                                  batch_size   =args.batch_size,
                                  sampler      =train_sampler,
                                  pin_memory   =True, 
                                  drop_last    =True,
                                  collate_fn   =lambda x: data_processing(x, "train"),
                                  num_workers  =args.num_workers)

        test_loader = DataLoader(dataset       =dataset2,
                                  batch_size   =args.test_batch_size,
                                  sampler      =test_sampler,
                                  pin_memory   =True, 
                                  drop_last    =True,
                                  collate_fn   =lambda x: data_processing(x, "valid"),
                                  num_workers  =args.num_workers)
        
    else:
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, collate_fn   =data_processing_train)
        test_loader  = torch.utils.data.DataLoader(dataset2, **test_kwargs , collate_fn   =data_processing_test)
        
    model = DeepSpeechNNET(n_cnn_layers=3, #3
                            n_rnn_layers=5, #5
                            rnn_dim=512, #512
                            n_class=29,
                            n_feats=128).to(device) ## 128

    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate/10)

    if args.hpu:
        permute_params(model, True, args.use_lazy_mode)
        permute_momentum(optimizer, True, args.use_lazy_mode)

    if args.distributed and args.hpu:
        model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=100, broadcast_buffers=False,
                gradient_as_bucket_view=True)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        if args.hpu:
            torch.save(model.cpu().state_dict(), "mnist_cnn.pt")
        else:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

