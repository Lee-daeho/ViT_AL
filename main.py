from cgi import test
import os
import random
import numpy as np

from collections import Counter
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

import torch

from data.data_utils import *
from models.transformers import *
from selection_methods import *


def val(args, model, test_loader, criterion):
    
    model.eval()
    val_acc = 0
    val_loss = 0

    for data, label in tqdm(test_loader):
        data = data.to(args.device)
        label = label.to(args.device)

        output = model(data)
        val_loss = criterion(output, label)

        acc = (output.argmax(dim=1) == label).float().mean()
        val_acc += acc / len(test_loader)
        val_loss += val_loss / len(test_loader)
    
    return val_acc, val_loss


def train(args, model, train_loader, test_loader, cycle, time):

    # Save results
    results = open(os.path.join("logs", args.name) + '/' + time + '_results_' + str(args.method_type) + '_' + args.dataset + '.txt', 'a')

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    t_total = args.num_steps

    # Train!

    model.zero_grad()
    #set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    global_step, best_acc = 0, 0
    
    for e in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(args.device)
            label = label.to(args.device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            if e % args.val_epoch == 0:
                val_accuracy, val_loss = val(args, model, test_loader, criterion)

        print() #############TODO###################


def main():
    args.device = device

    method = args.method_type
    time = datetime.now().strftime("%y-%m-%d-%H-%M")
    label_dist = open(os.path.join("logs", args.name) + '/' + time + '_' + str(args.method_type) + '_labels.txt', 'a')
    # Repeat for TRIAL times
    TRIAL = 5
    for trial in range(TRIAL):
    # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                    (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

        #Loading training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset, args)
        ADDEDNUM = adden
        NUM_TRAIN = no_train
        SUBSET = 10000

        #Init model
        model = Vit(args.img_size, NO_CLASSES, args.patch_size, args.num_heads, args.hidden_size, args.num_layers, args.mlp_size)

        #make initial dataset
        indices = list(range(NUM_TRAIN))

        random.shuffle(indices)

        if args.total:
            labeled_set = indices
        else:
            labeled_set = indices[:ADDEDNUM]
            unlabeled_set = [x for x in indices if x not in labeled_set]
        
        train_loader = DataLoader(data_train,
                                    batch_size = args.train_batch_size,
                                    sampler=SubsetRandomSampler(labeled_set),
                                    pin_memory=True, 
                                    drop_last=False)
        
        test_loader = DataLoader(data_test, 
                                    batch_size=args.eval_batch_size,
                                    pin_memory=True)

        # Active Learning
        for cycle in range(args.cycles):

            # Randomly Sample 10000 unlabaled data points
            if not args.total:
                random.shuffle(unlabeled_set[:SUBSET])
                subset = unlabeled_set[:SUBSET]
            
            #to save the amount of data per labels
            c = Counter()
            for idx, (_, y) in enumerate(train_loader):
                c.update(y.numpy())

            
            np.array([args.method_type, 1000*(cycle+1), str(c)[7:]]).tofile(label_dist, sep=" ")
            label_dist.write('\n')
            train(args, model, train_loader, test_loader, cycle, time)

            arg = query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args)

            labeled_set += list(torch.tensor(subset)[arg][-ADDEDNUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy())
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))
            # Create a new dataloader for the updated labeled dataset
            train_loader = DataLoader(data_train, batch_size=args.batch_size,
                                                sampler=SubsetRandomSampler(labeled_set),
                                                pin_memory=True,
                                                drop_last=False)
            
                                

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="default", help="Name of experiment you would like to save")
    parser.add_argument("-d", "--dataset", type=str, choices=["cifar10", "cifar100", "imagenet", "svhn"], default="cifar10", help="Which dataset for")
    parser.add_argument("-m", "--method_type", type=str, default="lloss")
    parser.add_argument("-c", "--cycles", type=int, default=10, help="Number of active learning cycles")
    parser.add_argument("-t", "--total", type=bool, default=False, help="Training on the entire dataset")
    parser.add_argument("--trial", type=int, default=5, help="how many trials to repeat")
    parser.add_argument("-l", "--lambda_loss", type=float, default=1.2, help="Adjustment graph loss parameter between the labeled and unlabeled")
    parser.add_argument("-s", "--s_margin", type=float, default=0.1, help="Confidence margin of graph")
    parser.add_argument("-n", "--hidden_units", type=int, default=128,  help="Number of hidden units of the graph")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Transformer Training poch")
    parser.add_argument("--img_size", type=int, default=224, help="Resolution size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden vector size for Transformer")
    parser.add_argument("--mlp_size", type=int, default=3072, help="Hidden vector size for MLP")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of Encoder Layers")
    parser.add_argument("--learning_rate", type=float, default=3e-2, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--val_epoch", type=int, default=5, help="Number of epochs for validation")

    args = parser.parse_args()

    device = "cuda" if  torch.cuda.is_available() else "cpu"

    args.device = device

    main(args)

    