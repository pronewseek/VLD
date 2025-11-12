import torch.nn as nn
import torch
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    # try:
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # except:
    #     print('1')
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def l2norm(x):
    """L2-normalize columns of x"""
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)

def diversity_loss( x):
    x = l2norm(x)  # Columns of x MUST be l2-normalized
    gram_x = x.bmm(x.transpose(1, 2))
    I = torch.autograd.Variable(
        (torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1)
    )
    if torch.cuda.is_available():
        # I = I.cuda()
        I = I.to(device)
    gram_x.masked_fill_(I, 0.0)
    loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (x.size(1) ** 2)
    return loss.mean()


class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
                        i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss

class TripletLoss_WRT(nn.Module):

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        inputs = inputs.float()
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer."""

    def __init__(self, num_classes, epsilon=0.1, device=None):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class DynamicLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon_start=0.1, epsilon_end=0.05, device=None):
        super(DynamicLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, epoch_ratio=0):  # epoch_ratio: 当前epoch/总epoch数
        log_probs = self.logsoftmax(inputs)
        # 动态调整 epsilon
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * epoch_ratio

        targets_one_hot = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets_one_hot = targets_one_hot.to(self.device)
        targets_smooth = (1 - epsilon) * targets_one_hot + epsilon / self.num_classes
        loss = (- targets_smooth * log_probs).mean(0).sum()
        return loss

class ModalityAwareLabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, eps_rgb=0.05, eps_ir=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eps_rgb = eps_rgb
        self.eps_ir = eps_ir

    def update_eps(self, epoch, total_epoch):
        # decay = 1 - epoch / total_epoch
        # self.eps_rgb = self.eps_rgb * decay
        # self.eps_ir = self.eps_ir * decay

        # 方案1：设置最小值，避免完全衰减到0
        # min_eps_rgb = 0.01  # 最小eps值
        # min_eps_ir = 0.02
        # decay = 1 - epoch / total_epoch
        # self.eps_rgb = max(self.eps_rgb * decay, min_eps_rgb)
        # self.eps_ir = max(self.eps_ir * decay, min_eps_ir)

        # # 方案2：使用余弦衰减或其他平滑衰减策略
        # decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epoch))
        # self.eps_rgb = self.eps_rgb * decay
        # self.eps_ir = self.eps_ir * decay

        progress = epoch / total_epoch

        # 前期保持较高平滑度，后期逐渐降低
        if progress < 0.5:
            # 线性衰减
            decay = 1 - 0.5 * progress
        else:
            # 余弦衰减
            decay = 0.5 * (1 + math.cos(math.pi * (progress - 0.5) / 0.5))

        self.eps_rgb = self.eps_rgb * decay
        self.eps_ir = self.eps_ir * decay

    def forward(self, logits, targets, modalities):
        """
        logits: [B, C] 模型输出 (classification head)
        targets: [B] 真实类别标签 (person ID)
        modalities: [B] 模态标签 (RGB=0, IR=1)
        """
        B, C = logits.size()
        log_probs = F.log_softmax(logits, dim=1)

        # 生成每个样本的 ε 参数（模态感知）
        eps = torch.where(modalities == 0, self.eps_rgb, self.eps_ir).to(logits.device)
        # 动态调节平滑强度：置信度越高，平滑越小
        eps = eps.unsqueeze(1)  # [B,1]

        # 构建平滑后的标签分布
        smooth_targets = torch.full_like(log_probs, 0)  # 初始化全0矩阵
        smooth_targets.fill_(0)  # 可省略，明确逻辑

        # 每个样本填充 ε / (C - 1)
        smooth_targets = torch.ones_like(log_probs) * (eps / (C - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - eps)

        loss = (-smooth_targets * log_probs).sum(dim=1).mean()
        return loss







def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n) #B, B
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat) #B,B
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss









