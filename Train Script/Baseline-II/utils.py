############################# Logging function #############################
import logging
import logging.handlers

logger = logging.getLogger(__name__)

class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s: [%(levelname)s] - %(message)s')
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)


############################# Get Dataset function #############################
from torchvision.transforms import v2
from torchvision import datasets

def get_dataset(args):
    if args.dataset.lower() == 'cifar10':
        transform_train = v2.Compose([
            v2.PILToTensor(),
            v2.Pad(4, padding_mode='edge'),
            v2.RandomCrop(32),
            v2.Resize((128, 128)),
            v2.RandomHorizontalFlip(0.5),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        transform_test = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((128, 128)),
            v2.ToDtype(torch.float32, scale=True),
        ])

        Train_data = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform_train)
        Test_data = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform_test)
        return Train_data, Test_data
    elif args.dataset.lower() == 'cifar100':
        transform_train = v2.Compose([
            v2.PILToTensor(),
            v2.Pad(4, padding_mode='edge'),
            v2.RandomCrop(32),
            v2.Resize((128, 128)),
            v2.RandomHorizontalFlip(0.5),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        transform_test = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((128, 128)),
            v2.ToDtype(torch.float32, scale=True),
        ])

        Train_data = datasets.CIFAR100(root=args.dataset_path, train=True, download=True, transform=transform_train)
        Test_data = datasets.CIFAR100(root=args.dataset_path, train=False, download=True, transform=transform_test)
        return Train_data, Test_data
    elif args.dataset.lower() == 'imagenet':
        transform_train = v2.Compose([
            v2.PILToTensor(),
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(0.5),
            v2.ToDtype(torch.float32, scale=True),
        ])

        transform_test = v2.Compose([
            v2.PILToTensor(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
        ])

        Train_data = datasets.ImageFolder(root=args.dataset_path+'train', transform=transform_train)
        Test_data = datasets.ImageFolder(root=args.dataset_path+'val', transform=transform_test)
        return Train_data, Test_data


########################################### Get info ##########################
import os
import csv
import torch
import shutil
from enum import Enum
import torch.distributed as dist

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, epoch, weights_path, acc1, acc5, losses):
    filename = weights_path + f'epoch{epoch}.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, weights_path+f'best.pth.tar')
    if os.path.exists(weights_path + f'epoch{epoch-1}.pth.tar'):
        os.remove(weights_path + f'epoch{epoch-1}.pth.tar')

    with open(weights_path+f'record.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        data = [[epoch, acc1, acc5, losses]]
        writer.writerows(data)