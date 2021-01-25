import torch

def rpn_dict(rpn_locs, rpn_scores, rois, roi_indices, anchor):
    vardict= {}
    vardict["rpn_locs"] =rpn_locs
    vardict["rpn_scores"] = rpn_scores
    vardict["rois"] = rois
    vardict["roi_indices"] = roi_indices
    vardict["anchor"] = anchor

    return vardict