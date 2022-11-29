# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch.optim.optimizer import Optimizer, required

# %%
# per paper - this gets pretrained with the contrastive loss, then we train a linear classifier with CE loss on top of it
class ContrastiveCellTypeEncoder(nn.Module):
    
    def __init__(self, emb_dim = 512, feat_dim = 128):
        
        super().__init__()

        # encoder - convnet backbone mapping image to latent features in R^512 that lie on unit hypersphere (normalized so embeddings have length 1)
        model = torchvision.models.resnet18()
        trained_kernel = model.conv1.weight
        new_conv = nn.Conv2d(6, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        with torch.no_grad():
            new_conv.weight[:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
        model.conv1 = new_conv
        self.features = nn.ModuleList(model.children())[:-1]
        self.features = nn.Sequential(*self.features)

        # projection network - maps embeddings to a 128-dimensional vector of length 1 (lies on unit hypersphere - allows for dot products to show similarity)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace = True),
            nn.Linear(emb_dim, feat_dim)
        )

    def forward(self, x):
        
        x = self.features(x).flatten(1)
        encoder_output = F.normalize(F.relu(x), dim = 1)
        projection = self.head(encoder_output)
        projection = F.normalize(projection, dim = 1)

        try:
            assert torch.allclose(torch.linalg.norm(projection, dim = 1), torch.tensor(1.0))
        except:
            print(projection)
            print(projection.shape)

        return projection, encoder_output

class ContrastiveCellTypeClassifier(nn.Module):

    def __init__(self, emb_dim = 512, n_classes = 1139):
        
        super().__init__()
        self.embedding = nn.Embedding(4, emb_dim)
        self.fc = nn.Linear(emb_dim, n_classes)

    def forward(self, x, cell_types):

        emb = self.embedding(cell_types)
        emb_x = x * emb
        output = F.relu(emb_x)
        output = self.fc(output)

        return output


# %%
# https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class SupConLoss(nn.Module):
    
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# %%
# https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
class LARS(Optimizer):
    """Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    Example:
        optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        optimizer.zero_grad()
        loss_fn(model(input), target).backward()
        optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=.9,
                 weight_decay=.0005, eta=0.001, max_epoch=200):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta, max_epoch=max_epoch)
        super(LARS, self).__init__(params, defaults)

    def step(self, epoch=None, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            max_epoch = group['max_epoch']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # Global LR computed on polynomial decay schedule
                decay = (1 - float(epoch) / max_epoch) ** 2
                global_lr = lr * decay

                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / \
                    (grad_norm + weight_decay * weight_norm)

                # Update the momentum term
                actual_lr = local_lr * global_lr

                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = \
                            torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(actual_lr, d_p + weight_decay * p.data)
                p.data.add_(-buf)

        return loss