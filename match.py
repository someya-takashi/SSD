import torch


# DBoxの情報をBBox形式に変換する関数

def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), 1)

# 2個のボックスが重なる部分の面積を求める関数

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)

    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2)
    )

    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2)
    )

    inter = torch.clamp((max_xy - min_xy), min = 0)

    return inter[:, :, 0] * inter[:, :, 1]

# ジャッカード係数（IoU)を求める関数

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)

    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(1).expand_as(inter)

    union = area_a + area_b - inter

    return inter / union


# 教師データloc, confを作成する関数

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(truths, point_form(priors))

    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]

    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0

    loc = encode(matches, priors, variances)
    
    loc_t[idx] = loc
    conf_t[idx] = conf
    

# DBoxのオフセット情報を作る関数

def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] - matched[:, :2]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], 1)

