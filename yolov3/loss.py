import torch
from torch import nn
from yolov3.utils.boxes_op import bbox_iou
def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
def norm_location(tensor, img_size):
    if (not isinstance(img_size, tuple)) and (not isinstance(img_size, list)):
        img_size = [img_size, img_size]
    w = (tensor[:, 2] - tensor[:, 0]).unsqueeze(1) / img_size[0]
    h = (tensor[:, 3] - tensor[:, 1]).unsqueeze(1) / img_size[1]
    xc = (tensor[:, 2] + tensor[:, 0]).unsqueeze(1) / (img_size[0] * 2)
    yc = (tensor[:, 3] + tensor[:, 1]).unsqueeze(1) / (img_size[1] * 2)
    return torch.cat([xc, yc, w, h], dim=1)


class YOLOLossComputation(object):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.red = cfg.MODEL.YOLOV3.LOSS_REDUCE
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.cuda.FloatTensor([cfg.MODEL.YOLOV3.CLS_WGT]),
                                           reduction=self.red)
        self.Objcls = nn.BCEWithLogitsLoss(pos_weight=torch.cuda.FloatTensor([cfg.MODEL.YOLOV3.OBJ_WGT]),
                                           reduction=self.red)
        self.smooth_pos, self.smooth_neg = smooth_BCE(eps=cfg.MODEL.YOLOV3.SMOOTH_EPS)
        self.model = model

        if cfg.MODEL.YOLOV3.FL_WGT >0.:
            self.BCEcls = FocalLoss(self.BCEcls, cfg.MODEL.YOLOV3.FL_GAMMA,
                                                 cfg.MODEL.YOLOV3.FL_ALPHA)
            self.Objclss = FocalLoss(self.Objcls, cfg.MODEL.YOLOV3.FL_GAMMA,
                                                 cfg.MODEL.YOLOV3.FL_ALPHA)

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        nt = targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
        off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

        style = None
        for i, j in enumerate(self.model.yolo_layers):
            anchors = self.model.module_list[j].anchor_vec
            gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            na = anchors.shape[0]  # number of anchors
            at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

            # Match targets to anchors
            a, t, offsets = [], targets * gain, 0
            if nt:
                # r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
                # j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
                j = wh_iou(anchors, t[:, 4:6]) > self.cfg.MODEL.YOLOV3.IOU_TH # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

                # overlaps
                gxy = t[:, 2:4]  # grid xy
                z = torch.zeros_like(gxy)
                if style == 'rect2':
                    g = 0.2  # offset
                    j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                    a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                    offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

                elif style == 'rect4':
                    g = 0.5  # offset
                    j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                    l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                    a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                    offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            if c.shape[0]:  # if any targets
                assert c.max() < self.cfg.MODEL.YOLOV3.NUM_CLASSES, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                           'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                               self.cfg.MODEL.YOLOV3.NUM_CLASSES,
                                               self.cfg.MODEL.YOLOV3.NUM_CLASSES - 1, c.max())

        return tcls, tbox, indices, anch

    def __call__(self, pred, target):
        imgIdx = []
        bboxes = []
        labels = []
        for im_i in range(len(target)):
            imgIdx.append(torch.full(size=(len(target[im_i]),1),
                                     fill_value=im_i,
                                     dtype=torch.int,
                                     device=pred[0].device))
            bboxes.append(norm_location(target[im_i].gt_boxes.tensor.to(pred[0].device),
                                        target[im_i].image_size))
            labels.append(target[im_i].gt_classes.to(pred[0].device).unsqueeze(1))
        imgIdx = torch.cat(imgIdx, dim=0).float()
        bboxes = torch.cat(bboxes, dim=0).float()
        labels = torch.cat(labels, dim=0).float()

        yolo_target = torch.cat([imgIdx, labels, bboxes], dim=1)

        ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
        lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
        tcls, tbox, indices, anchors = self.build_targets(pred, yolo_target)
        nt = 0
        for i, pred_i in enumerate(pred):  # layer index, layer predictions
            batch_idx, anchor_idx, grid_j, grid_i = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pred_i[..., 0])  # target obj

            num_batch = batch_idx.shape[0]  # number of targets
            if num_batch:
                nt += num_batch  # cumulative targets
                ps = pred_i[batch_idx, anchor_idx, grid_j, grid_i]  # prediction subset corresponding to targets

                # GIoU
                pxy = ps[:, :2].sigmoid()
                pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
                lbox += (1.0 - giou).sum() if self.red == 'sum' else (1.0 - giou).mean()  # giou loss

                # Obj
                tobj[batch_idx, anchor_idx, grid_j, grid_i] \
                    = (1.0 - self.cfg.MODEL.YOLOV3.GIOU_RATIO) + \
                      self.cfg.MODEL.YOLOV3.GIOU_RATIO * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

                # Class
                if self.cfg.MODEL.YOLOV3.NUM_CLASSES > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.smooth_neg)  # targets
                    t[range(num_batch), tcls[i]] = self.smooth_pos
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            lobj += self.Objcls(pred_i[..., 4], tobj)  # obj loss

        lbox *= self.cfg.MODEL.YOLOV3.LOSS_GIOU_WGT
        lobj *= self.cfg.MODEL.YOLOV3.LOSS_OBJ_WGT
        lcls *= self.cfg.MODEL.YOLOV3.LOSS_CLS_WGT
        if self.red == 'sum':
            bs = tobj.shape[0]  # batch size
            g = 3.0  # loss gain
            lobj *= g / bs
            if nt:
                lcls *= g / nt / self.cfg.MODEL.YOLOV3.NUM_CLASSES
                lbox *= g / nt

        # loss = lbox + lobj + lcls
        # return loss, torch.cat((lbox, lobj, lcls, loss)).detach()
        return {'loss_box': lbox, 'loss_obj': lobj, 'loss_cls': lcls}