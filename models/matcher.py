# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        """
We did try a few fixed orderings, including arbitrary fixed order (eg the dataset order), as well as a few lexicographical orders (sorting by position, size, class, ...), but all these attempts performed worse.

If you enforce an ordering, you are essentially putting the additional constraint that the network must output the "true" predictions in the first queries, and then all predict "no-object" for all the remaining queries. Say if there are 100 object queries, which is the default in DETR, but only 3 objects to detect, then predictions [1, 3] must match the targets (according to whichever order you chose), and predictions [4, 100] must be "no objects".
By contrast, with the Hungarian matching, it doesn't matter where the "true" predictions are, they can be scattered anywhere amongst the predictions. In my example, the network can decide to use queries say 43, 57 and 99 to predict the objects, and fill the rest with "no-object".

Here is some intuition why this works better:

Since the network doesn't have to push the predictions to the beginning, it can instead let each query specialize in its own kind of objects. We show in the paper that each object query tend to predict objects in a specific region of the image, which would not be possible with a forced ordering.
Related to the previous point, it's rather clear that the fixed ordering is a worse usage of the queries. In coco for ex, there is no image with more than ~75 objects. With a fixed ordering, it means that the object queries in [75-100] will never be used. Similarly, there are few images with more than >50 objects, so queries [50-75] could potentially overfit to the said images. By contrast, with the Hungarian matching, increasing the number of queries improve the recall and thus the AP.
Finally, Hungarian matching is more robust to noise in the annotations. Say for example there are 3 objects visible A, B and C, but for some reason only B and C are annotated. In the fixed ordering case, if the model predicts A, B, C (which would be theoretically correct), since the fixed ordering loss expects B, C, "no-object" it means that the model will be hugely penalized (none of the predictions will be correct). By contrast, with a Hungarian loss, B and C will be correctly reinforced, and the network will suffer only a small classification penalty for predicting A.
I hope this helps giving more intuition about that.
I believe I have answered your question, and as such I'm closing this issue. Feel free to reach out if you have further concerns.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # [batch_size * tgt_num]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # [batch_size * tgt_num, 4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]
        # shape: batch_size * num_queries, tgt_num

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # shape: batch_size * num_queries, tgt_num

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        # shape: batch_size * num_queries, tgt_num

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        # shape: batch_size, num_queries, tgt_num

        sizes = [len(v["boxes"]) for v in targets]  # shape: batch_size
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # 对最好的query进行优化
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
