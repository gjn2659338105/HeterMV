import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, config, args):
        super(Classifier, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_labels = args.num_labels
        self.current_device = args.device
        self.softmax = nn.Softmax(dim=-1)
        self.lambda_ctx = getattr(args, "lambda_ctx", 0.1)
        self.lambda_ref = getattr(args, "lambda_ref", 0.2)

    def forward(self, claim_emb_list, fusion_emb, labels, 
                evid_only_emb=None, evid_ctx_emb=None, evid_ref_emb=None):
        logits = []
        for label_id in range(len(claim_emb_list)):
            logits_one_label = torch.sum(torch.multiply(claim_emb_list[label_id], fusion_emb), dim=-1)
            logits.append(torch.reshape(logits_one_label, [-1, 1]))
        logits = torch.concat(logits, dim=-1)

        y_pred = torch.argmax(logits, dim=-1)
        one_hot = nn.functional.one_hot(labels.to(self.current_device), num_classes=self.num_labels).float()
        y_pred_prob = self.softmax(logits)
        y_pred_prob = torch.clamp(y_pred_prob, min=1e-12)
        loss_cls = - torch.sum(one_hot * torch.log(y_pred_prob), dim=-1).mean()

        # ------ 对比损失 ------
        loss_ctx = self.info_nce_loss(evid_only_emb, evid_ctx_emb) if evid_ctx_emb is not None else 0.0
        loss_ref = self.info_nce_loss(evid_only_emb, evid_ref_emb) if evid_ref_emb is not None else 0.0

        loss_total = loss_cls + self.lambda_ctx * loss_ctx + self.lambda_ref * loss_ref
        # If you want to do ablation study
        # loss_total = loss_cls + self.lambda_ctx * loss_ctx
        # loss_total = loss_cls + self.lambda_ref * loss_ref
        return loss_total, y_pred
        
        # return loss_cls, y_pred

    @staticmethod
    def info_nce_loss(z1, z2, temperature=0.1):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(logits, labels)
