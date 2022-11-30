import math
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor

@ops.constexpr
def get_tensor(x, dtype=ms.float32):
    return Tensor(x, dtype)

@ops.constexpr(reuse_result=True)
def get_pi(dtype=ms.float32):
    return Tensor(math.pi, dtype)

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        x1, y1, w1, h1 = ops.split(box1, 1, 4)
        x2, y2, w2, h2 = ops.split(box2, 1, 4)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = ops.split(box1, 1, 4)
        b2_x1, b2_y1, b2_x2, b2_y2 = ops.split(box2, 1, 4)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0., None) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0., None)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(b1_x1, b2_x1) # convex (smallest enclosing box) width
        ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / get_pi(iou.dtype) ** 2) * ops.pow(ops.atan(w2 / (h2 + eps)) - ops.atan(w1 / (h1 + eps)), 2)
                alpha = v / (v - iou + (1 + eps))
                alpha = ops.stop_gradient(alpha)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class BCEWithLogitsLoss(nn.Cell):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, bce_weight=None, bce_pos_weight=None, reduction="mean"):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.reduction = reduction # default mean
        assert self.loss_fcn.reduction == 'none'  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        ori_dtype = pred.dtype
        loss = self.loss_fcn(pred.astype(ms.float32), true.astype(ms.float32))

        if mask is not None:
            loss *= mask

        if self.reduction == 'mean':
            if mask is not None:
                return (loss.sum() / mask.astype(loss.dtype).sum().clip(1, None)).astype(ori_dtype)
            return loss.mean().astype(ori_dtype)
        elif self.reduction == 'sum':
            return loss.sum().astype(ori_dtype)
        else:  # 'none'
            return loss.astype(ori_dtype)
        
class FocalLoss(nn.Cell):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, bce_weight=None, bce_pos_weight=None, gamma=1.5, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction # default mean
        assert self.loss_fcn.reduction == 'none'  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        ori_dtype = pred.dtype
        loss = self.loss_fcn(pred.astype(ms.float32), true.astype(ms.float32))
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = ops.Sigmoid()(pred) # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if mask is not None:
            loss *= mask

        if self.reduction == 'mean':
            if mask is not None:
                return (loss.sum() / mask.astype(loss.dtype).sum().clip(1, None)).astype(ori_dtype)
            return loss.mean().astype(ori_dtype)
        elif self.reduction == 'sum':
            return loss.sum().astype(ori_dtype)
        else:  # 'none'
            return loss.astype(ori_dtype)

# class ComputeLoss(nn.Cell):
#     # Compute losses
#     def __init__(self, model, autobalance=False):
#         super(ComputeLoss, self).__init__()
#         self.sort_obj_iou = False

#         h = model.hyp  # hyperparameters
#         self.hyp_anchor_t = h["anchor_t"]
#         self.hyp_box = h['box']
#         self.hyp_obj = h['obj']
#         self.hyp_cls = h['cls']

#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h['cls_pw']], ms.float32), gamma=g),\
#                              FocalLoss(bce_pos_weight=Tensor([h['obj_pw']], ms.float32), gamma=g)
#         else:
#             # Define criteria
#             BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['cls_pw']]), ms.float32))
#             BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['obj_pw']]), ms.float32))

#         m = model.model[-1]  # Detect() module
#         _balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#         self.balance = ms.Parameter(Tensor(_balance, ms.float32), requires_grad=False)
#         self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance
#         self.na = m.na  # number of anchors
#         self.nc = m.nc  # number of classes
#         self.nl = m.nl  # number of layers
#         self.anchors = m.anchors

#         self._off = Tensor([
#             [0, 0],
#             [1, 0],
#             [0, 1],
#             [-1, 0],
#             [0, -1],  # j,k,l,m
#             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#         ], dtype=ms.float32)

#     def construct(self, p, targets):  # predictions, targets
#         lcls, lbox, lobj = 0., 0., 0.

#         tcls, tbox, indices, anchors, tmasks = self.build_targets(p, targets)  # class, box, (image, anchor, gridj, gridi), anchors, mask
#         tcls, tbox, indices, anchors, tmasks = ops.stop_gradient(tcls), ops.stop_gradient(tbox), \
#                                                ops.stop_gradient(indices), ops.stop_gradient(anchors), \
#                                                ops.stop_gradient(tmasks)

#         # Losses
#         for layer_index, pi in enumerate(p):  # layer index, layer predictions
#             tmask = tmasks[layer_index]
#             b, a, gj, gi = ops.split(indices[layer_index] * tmask[None, :], 0, 4)  # image, anchor, gridy, gridx
#             b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)
#             tobj = ops.zeros(pi.shape[:4], pi.dtype) # target obj

#             n = b.shape[0]  # number of targets
#             if n:
#                 _meta_pred = pi[b, a, gj, gi] #gather from (bs,na,h,w,nc)
#                 pxy, pwh, _, pcls = _meta_pred[:, :2], _meta_pred[:, 2:4], _meta_pred[:, 4:5], _meta_pred[:, 5:]

#                 # Regression
#                 pxy = ops.Sigmoid()(pxy) * 2 - 0.5
#                 pwh = (ops.Sigmoid()(pwh) * 2) ** 2 * anchors[layer_index]
#                 pbox = ops.concat((pxy, pwh), 1)  # predicted box
#                 iou = bbox_iou(pbox, tbox[layer_index], CIoU=True).squeeze()  # iou(prediction, target)
#                 # iou = iou * tmask
#                 # lbox += (1.0 - iou).mean()  # iou loss
#                 lbox += ((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum()  # iou loss

#                 # Objectness
#                 iou = ops.Identity()(iou).clip(0, None)
#                 if self.sort_obj_iou:
#                     _, j = ops.sort(iou)
#                     b, a, gj, gi, iou, tmask = b[j], a[j], gj[j], gi[j], iou[j], tmask[j]
#                 if self.gr < 1:
#                     iou = (1.0 - self.gr) + self.gr * iou
#                 # tobj[b, a, gj, gi] = iou * tmask  # iou ratio
#                 tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.identity(iou).clip(0, None)) * tmask  # iou ratio

#                 # Classification
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = ops.fill(pcls.dtype, pcls.shape, self.cn) # targets

#                     t[mnp.arange(n), tcls[layer_index]] = self.cp
#                     # lcls += self.BCEcls(pcls, t, tmask[:, None])  # BCE
#                     lcls += self.BCEcls(pcls, t, ops.tile(tmask[:, None], (1, t.shape[-1])))

#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[layer_index]  # obj loss
#             if self.autobalance:
#                 self.balance[layer_index] = self.balance[layer_index] * 0.9999 + 0.0001 / obji.item()

#         if self.autobalance:
#             _balance_ssi = self.balance[self.ssi]
#             self.balance /= _balance_ssi
#         lbox *= self.hyp_box
#         lobj *= self.hyp_obj
#         lcls *= self.hyp_cls
#         bs = p[0].shape[0]  # batch size

#         loss = lbox + lobj + lcls

#         return loss * bs, ops.identity(ops.stack((lbox, lobj, lcls, loss)))

#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         targets = targets.view(-1, 6) # targets原shape(bs, target_num, (image, cls, x, y, w, h))
#         mask_t = targets[:, 1] >= 0
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch, tmasks = (), (), (), (), ()
#         gain = ops.ones(7, ms.int32) # normalized to gridspace gain
#         ai = ops.tile(mnp.arange(na).view(-1, 1), (1, nt)) # shape: (na, nt)
#         ai = ops.cast(ai, targets.dtype)
#         targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2) # append anchor indices # shape: (na, nt, 7)

#         g = 0.5  # bias
#         off = ops.cast(self._off, targets.dtype) * g  # offsets

#         for i in range(self.nl):
#             anchors, shape = self.anchors[i], p[i].shape
#             gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]] # xyxy gain

#             # Match targets to anchors
#             t = targets * gain  # shape(na,nt,7) # xywhn -> xywh
#             # Matches
#             # if nt:
#             r = t[..., 4:6] / anchors[:, None]  # wh ratio
#             j = ops.maximum(r, 1 / r).max(2) < self.hyp_anchor_t # compare / j_shape(na, nt)

#             # t = t[j]  # filter
#             mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1)
#             t = t.view(-1, 7)

#             # Offsets
#             gxy = t[:, 2:4]  # grid xy # grid xy targets中心点所在grid
#             gxi = gain[[2, 3]] - gxy  # inverse

#             jk = ops.logical_and((gxy % 1 < g), (gxy > 1)) #.astype(ms.int32)
#             lm = ops.logical_and((gxi % 1 < g), (gxi > 1)) #.astype(ms.int32)
#             j, k = jk[:, 0], jk[:, 1]
#             l, m = lm[:, 0], lm[:, 1] # 最终得到的j表示x坐标偏左的targets，k表示y坐标偏左的targets，l表示x坐标偏右的targets，m表示y坐标偏右的targets

#             # # Original
#             # j = ops.stack((ops.ones_like(j), j, k, l, m)) # shape: (5, *)
#             # t = ops.tile(t, (5, 1, 1)) # shape(5, *, 7)
#             # t = t.view(-1, 7)
#             # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
#             # # t = t.repeat((5, 1, 1))[j]
#             # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :]) #(1,*,2) + (5,1,2) -> (5,*,2)
#             # offsets = offsets.view(-1, 2)

#             # faster,
#             tag1, tag2 = ops.identity(j), ops.identity(k)
#             tag1, tag2 = ops.tile(tag1[:, None], (1, 2)), ops.tile(tag2[:, None], (1, 2))
#             j_l = ops.logical_or(j, l).astype(ms.int32)
#             k_m = ops.logical_or(k, m).astype(ms.int32)
#             center = ops.ones_like(j_l)

#             j = ops.stack((center, j_l, k_m))
#             t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
#             t = t.view(-1, 7)
#             mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
#             offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
#             offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
#             # offsets_new[0, :, :] = offsets[0, :, :]
#             offsets_new[1:2, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, :, :], offsets[3, :, :])
#             offsets_new[2:3, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, :, :], offsets[4, :, :])
#             offsets = offsets_new
#             offsets = offsets.view(-1, 2)

#             # Define
#             b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
#                                 ops.cast(t[:, 1], ms.int32), \
#                                 t[:, 2:4], \
#                                 t[:, 4:6], \
#                                 ops.cast(t[:, 6], ms.int32) # (image, class), grid xy, grid wh, anchors
#             gij = ops.cast(gxy - offsets, ms.int32)
#             gi, gj = gij[:, 0], gij[:, 1]  # grid indices
#             gi = gi.clip(0, shape[3] - 1)
#             gj = gj.clip(0, shape[2] - 1)


#             # Append
#             indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
#             tbox += (ops.concat((gxy - gij, gwh), 1),)  # box
#             anch += (anchors[a],)  # anchors
#             tcls += (c,)  # class
#             tmasks += (mask_m_t,)

#         return ops.stack(tcls), \
#                ops.stack(tbox), \
#                ops.stack(indices), \
#                ops.stack(anch), \
#                ops.stack(tmasks) # class, box, (image, anchor, gridj, gridi), anchors, mask

class ComputeLoss(nn.Cell):
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False

        h = model.hyp  # hyperparameters
        self.hyp_anchor_t = h["anchor_t"]
        self.hyp_box = h['box']
        self.hyp_obj = h['obj']
        self.hyp_cls = h['cls']

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h['cls_pw']], ms.float32), gamma=g),\
                             FocalLoss(bce_pos_weight=Tensor([h['obj_pw']], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['cls_pw']]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['obj_pw']]), ms.float32))

        m = model.model[-1]  # Detect() module
        _balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance = ms.Parameter(Tensor(_balance, ms.float32), requires_grad=False)
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors

        self._off = Tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ], dtype=ms.float32)

    def construct(self, p, targets):  # predictions, targets
        lcls, lbox, lobj = 0., 0., 0.

        tcls, tbox, indices, anchors, tmasks = self.build_targets(p, targets)  # class, box, (image, anchor, gridj, gridi), anchors, mask
        tcls, tbox, indices, anchors, tmasks = ops.stop_gradient(tcls), ops.stop_gradient(tbox), \
                                               ops.stop_gradient(indices), ops.stop_gradient(anchors), \
                                               ops.stop_gradient(tmasks)

        # Losses
        for layer_index, pi in enumerate(p):  # layer index, layer predictions
            tmask = tmasks[layer_index]
            b, a, gj, gi = ops.split(indices[layer_index] * tmask[None, :], 0, 4)  # image, anchor, gridy, gridx
            b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)
            tobj = ops.zeros(pi.shape[:4], pi.dtype) # target obj

            n = b.shape[0]  # number of targets
            if n:
                _meta_pred = pi[b, a, gj, gi] #gather from (bs,na,h,w,nc)
                pxy, pwh, _, pcls = _meta_pred[:, :2], _meta_pred[:, 2:4], _meta_pred[:, 4:5], _meta_pred[:, 5:]

                # Regression
                pxy = ops.Sigmoid()(pxy) * 2 - 0.5
                pwh = (ops.Sigmoid()(pwh) * 2) ** 2 * anchors[layer_index]
                pbox = ops.concat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[layer_index], CIoU=True).squeeze()  # iou(prediction, target)
                # iou = iou * tmask
                # lbox += (1.0 - iou).mean()  # iou loss
                lbox += ((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum()  # iou loss

                # Objectness
                iou = ops.Identity()(iou).clip(0, None)
                if self.sort_obj_iou:
                    _, j = ops.sort(iou)
                    b, a, gj, gi, iou, tmask = b[j], a[j], gj[j], gi[j], iou[j], tmask[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                # tobj[b, a, gj, gi] = iou * tmask  # iou ratio
                tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.identity(iou).clip(0, None)) * tmask  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = ops.fill(pcls.dtype, pcls.shape, self.cn) # targets

                    t[mnp.arange(n), tcls[layer_index]] = self.cp
                    lcls += self.BCEcls(pcls, t, ops.tile(tmask[:, None], (1, t.shape[-1])))  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[layer_index]  # obj loss
            if self.autobalance:
                self.balance[layer_index] = self.balance[layer_index] * 0.9999 + 0.0001 / obji.item()

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls

        return loss * bs, ops.identity(ops.stack((lbox, lobj, lcls, loss)))

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6)
        mask_t = targets[:, 1] >= 0
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tmasks = (), (), (), (), ()
        gain = ops.ones(7, ms.int32) # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na).view(-1, 1), (1, nt)) # shape: (na, nt)
        ai = ops.cast(ai, targets.dtype)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2) # append anchor indices # shape: (na, nt, 7)

        g = 0.5  # bias
        off = ops.cast(self._off, targets.dtype) * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]] # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(na,nt,7) # xywhn -> xywh
            # Matches
            # if nt:
            r = t[..., 4:6] / anchors[:, None]  # wh ratio
            j = ops.maximum(r, 1 / r).max(2) < self.hyp_anchor_t # compare

            # t = t[j]  # filter
            mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1)
            t = t.view(-1, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            jk = ops.logical_and((gxy % 1 < g), (gxy > 1)) #.astype(ms.int32)
            lm = ops.logical_and((gxi % 1 < g), (gxi > 1)) #.astype(ms.int32)
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # # Original
            # j = ops.stack((ops.ones_like(j), j, k, l, m)) # shape: (5, *)
            # t = ops.tile(t, (5, 1, 1)) # shape(5, *, 7)
            # t = t.view(-1, 7)
            # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # # t = t.repeat((5, 1, 1))[j]
            # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :]) #(1,*,2) + (5,1,2) -> (5,*,2)
            # offsets = offsets.view(-1, 2)

            # faster,
            tag1, tag2 = ops.identity(j), ops.identity(k)
            tag1, tag2 = ops.tile(tag1[:, None], (1, 2)), ops.tile(tag2[:, None], (1, 2))
            j_l = ops.logical_or(j, l).astype(ms.int32)
            k_m = ops.logical_or(k, m).astype(ms.int32)
            center = ops.ones_like(j_l)
            j = ops.stack((center, j_l, k_m))
            t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
            # offsets_new[0, :, :] = offsets[0, :, :]
            offsets_new[1:2, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, :, :], offsets[3, :, :])
            offsets_new[2:3, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, :, :], offsets[4, :, :])
            offsets = offsets_new
            offsets = offsets.view(-1, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32) # (image, class), grid xy, grid wh, anchors
            gij = ops.cast(gxy - offsets, ms.int32)
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)


            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            tbox += (ops.concat((gxy - gij, gwh), 1),)  # box
            anch += (anchors[a],)  # anchors
            tcls += (c,)  # class
            tmasks += (mask_m_t,)

        return ops.stack(tcls), \
               ops.stack(tbox), \
               ops.stack(indices), \
               ops.stack(anch), \
               ops.stack(tmasks) # class, box, (image, anchor, gridj, gridi), anchors, mask