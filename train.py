# train.py
import os
import argparse
import logging
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import Net, initWeights
from data import MyDataset, AugData, ToTensor

# Training configurations
IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
EPOCHS = 200
LR = 0.001
WEIGHT_DECAY = 5e-4
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
OUTPUT_PATH = Path(__file__).stem


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)
        data, label = sample['data'], sample['label']
        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)
        data, label = data.to(device), label.to(device)

        if i == 0 and epoch == 1:
            logging.info(f"Input data mean: {data.mean().item():.4f}, std: {data.std().item():.4f}")

        optimizer.zero_grad()
        end = time.time()
        output = model(data, debug=True, epoch=epoch, batch_idx=i)

        if i == 0 and epoch == 1:
            probs = torch.softmax(output, dim=1)
            logging.info(f"Output probabilities (first sample): {probs[0].detach().cpu().numpy()}")

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % TRAIN_PRINT_FREQUENCY == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']
            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    accuracy = correct / (len(eval_loader.dataset) * 2)
    logging.info('-' * 8)
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return accuracy


def setLogger(log_path, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def main(args):
    statePath = args.statePath
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("CUDA is not available. Using CPU instead.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_transform = transforms.Compose([ToTensor()])
    eval_transform = transforms.Compose([ToTensor()])

    TRAIN_DATASET_DIR = r"D:\python_project\yenet\data\train"
    VALID_DATASET_DIR = r"D:\python_project\yenet\data\val"
    TEST_DATASET_DIR = r"D:\python_project\yenet\data\test"

    Model_NAME = 'STEGO_Suniward_P0.2'
    info = 'AUG'
    output_dir = os.path.join(OUTPUT_PATH, f"{Model_NAME}_{info}")
    PARAMS_NAME = 'model_params.pt'
    LOG_NAME = 'model_log'

    os.makedirs(output_dir, exist_ok=True)

    PARAMS_PATH = os.path.join(output_dir, PARAMS_NAME)
    LOG_PATH = os.path.join(output_dir, LOG_NAME)
    setLogger(LOG_PATH, mode='w')

    train_dataset = MyDataset(TRAIN_DATASET_DIR, mode='train', transform=train_transform)
    valid_dataset = MyDataset(VALID_DATASET_DIR, mode='val', transform=eval_transform)
    test_dataset = MyDataset(TEST_DATASET_DIR, mode='test', transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # 实验 1：使用 SRM 滤波器
    logging.info("Training with SRM...")
    model_with_srm = Net(use_srm=True).to(device)
    model_with_srm.apply(initWeights)
    optimizer = optim.Adam(model_with_srm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-7)

    if statePath:
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(statePath))
        logging.info('-' * 8)
        all_state = torch.load(statePath)
        original_state = all_state['original_state']
        optimizer_state = all_state['optimizer_state']
        epoch = all_state['epoch']
        model_with_srm.load_state_dict(original_state)
        optimizer.load_state_dict(optimizer_state)
        startEpoch = epoch + 1
    else:
        startEpoch = 1

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    best_acc = 0.0
    patience = 10
    counter = 0
    for epoch in range(startEpoch, EPOCHS + 1):
        train(model_with_srm, device, train_loader, optimizer, epoch, scheduler)
        if epoch % EVAL_PRINT_FREQUENCY == 0:
            current_acc = evaluate(model_with_srm, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH)
            scheduler.step(current_acc)
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Current learning rate: {current_lr}")
            if current_acc > best_acc:
                best_acc = current_acc
                counter = 0
                torch.save({
                    'original_state': model_with_srm.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch
                }, PARAMS_PATH)
            else:
                counter += 1
            if epoch % 10 == 0:
                torch.save({
                    'original_state': model_with_srm.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch
                }, os.path.join(output_dir, f"model_with_srm_epoch_{epoch}.pt"))
            if counter >= patience:
                logging.info(f"Early stopping at epoch {epoch} due to no improvement in validation accuracy.")
                break

    # 实验 2：不使用 SRM 滤波器
    logging.info("Training without SRM...")
    model_without_srm = Net(use_srm=False).to(device)
    model_without_srm.apply(initWeights)
    optimizer = optim.Adam(model_without_srm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    best_acc = 0.0
    counter = 0
    for epoch in range(1, EPOCHS + 1):
        train(model_without_srm, device, train_loader, optimizer, epoch, scheduler)
        if epoch % EVAL_PRINT_FREQUENCY == 0:
            current_acc = evaluate(model_without_srm, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH)
            scheduler.step(current_acc)
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Current learning rate: {current_lr}")
            if current_acc > best_acc:
                best_acc = current_acc
                counter = 0
                torch.save({
                    'original_state': model_without_srm.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch
                }, os.path.join(output_dir, "model_without_srm.pt"))
            else:
                counter += 1
            if epoch % 10 == 0:
                torch.save({
                    'original_state': model_without_srm.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch
                }, os.path.join(output_dir, f"model_without_srm_epoch_{epoch}.pt"))
            if counter >= patience:
                logging.info(f"Early stopping at epoch {epoch} due to no improvement in validation accuracy.")
                break

    # 测试阶段：使用 SRM 的模型
    logging.info('\nTest set accuracy (with SRM): \n')
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model_with_srm.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)
    evaluate(model_with_srm, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH)


def myParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--statePath', help='Path for loading model state', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = myParseArgs()
    main(args)