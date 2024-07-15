import torch
import numpy as np


# referring segmentation metrics (mIoU, Pr@{0.5,0.6,0.7,0.8,0.9})
def segmentation_metrics(preds, masks, device="gpu"):
    iou_list = []
    for pred, mask in zip(preds, masks):
        # pred: (H, W): bool, mask: (H, W): bool
        # iou
        print(f"pred: \n{pred}\nmask: \n{mask}")
        inter = np.logical_and(pred, mask)
        union = np.logical_or(pred, mask)
        print(f"inter: \n{inter}\nunion: \n{union}")
        iou = np.sum(inter) / (np.sum(union) + 1e-6)
        print(f"iou: \n{iou}")
        iou_list.append(iou)
        inter_alt = pred[pred == mask]
        print(f"intersect alt: {inter_alt}")
        inter_alt_ = (pred == mask)
        print(f"intersect alt_: \n{inter_alt_}")
    print(f"iou list: \n{iou_list}")
    iou_list = np.stack(iou_list)
    print(f"iou list stacked : \n{iou_list}")
    iou_list = torch.from_numpy(iou_list).to(device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        print("tmp:")
        print(f"{(iou_list > thres)}")
        print(f"{(iou_list > thres).float()}")
        tmp = (iou_list > thres).float().mean()
        print(f"{tmp}\n")
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: IoU={:.2f}'.format(100. * iou.item())
    return head + temp, {'iou': iou.item(), **prec}

preds = []
masks = []
for i in range(5):
    pred = np.random.randint(5, size=(3, 4))
    mask = np.random.randint(5, size=(3, 4))
    preds.append(pred)
    masks.append(mask)
ret = segmentation_metrics(preds=preds, masks=masks, device='cpu')  
print(ret)   
# # eval function
# def eval_dataset(test_dataset, predict_fn):
#     metrics = {}
#     for data in test_dataset:
#         # print(refer_type)
#         all_preds, all_gt = [], []
#         for samp in tqdm(data):

#             img = samp['img']
#             pred = predict_fn(img)
            
#             all_preds.append(pred)
#             all_gt.append(samp['mask'])

#         met = segmentation_metrics(all_preds, all_gt, device="gpu")
#         print(met[0])
#         print()
#         metrics[samp['scene_id']] = met[1]
#     print(f'Average: {np.array([m["iou"] for m in metrics.values()]).mean()}')
#     return metrics


