import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM, BertForPreTraining
import torch.nn.functional as F
import math
from math import inf


class MMKD(nn.Module):
    """
    Build a MMKD model with a teacher encoder, a student encoder, and two MLPs
    """
    def __init__(self, dim=128, mlp_dim=768, T=0.05):
        """
        dim: feature dimension (default: 128)
        mlp_dim: hidden dimension in MLPs (default: 768)
        T: softmax temperature (default: 1.0)
        """
        super(MMKD, self).__init__()

        self.T = T
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        # build encoders: base_encoder indicates the student encoder, momentum_encoder indicates the teacher encoder.
        self.base_encoder = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        self.momentum_encoder = BertModel.from_pretrained('bert-base-cased')
        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        self.bert = self.base_encoder.bert
        self.cls = self.base_encoder.cls
        self.vocab_size = tokenizer.vocab_size
        for param_m in self.momentum_encoder.parameters():
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        # labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def byol_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        return 2-2*(q*k).sum(dim=-1)

    def compute_teacher_representations(self, input_ids, token_type_ids, attention_mask):
        bsz, seqlen = input_ids.size()
        outputs = self.momentum_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        rep = outputs[0]
        return rep

    def compute_representations(self, input_ids, token_type_ids, attention_mask):
        bsz, seqlen = input_ids.size()
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        rep = outputs[0]
        rep = rep.view(bsz,seqlen,768)
        return rep

    def compute_tlm_loss(self, truth, msk, logits, rep):
        truth = truth.transpose(0,1)
        msk = msk.transpose(0,1)
        y_tlm = logits.transpose(0,1).masked_select(msk.unsqueeze(-1).to(torch.bool))
        dump_y_tlm = y_tlm
        y_tlm = y_tlm.view(-1, self.vocab_size)
        gold = truth.masked_select(msk.to(torch.bool))
        log_probs_tlm = torch.log_softmax(y_tlm,-1)
        tlm_loss = F.nll_loss(log_probs_tlm, gold, reduction='mean')
        return tlm_loss

    def compute_tacl_loss(self, truth_rep, masked_rep, wlist_en, wlist_mul, contrastive_labels):
        bsz = len(wlist_en)
        loss_list = []
        num_seq = bsz
        flag = 0
        for i in range(bsz):
            num_word = len(set(wlist_en[i]))
            for j in range(num_word):
                if j==0:
                    word_en_rep = truth_rep[i,0,:].unsqueeze(0)
                    word_mul_rep = masked_rep[i,0,:].unsqueeze(0)
                    word_labels = contrastive_labels[i,0].unsqueeze(0)
                else:
                    token_index_en = wlist_en[i].index(j)
                    select_token_en = truth_rep[i,token_index_en,:]
                    token_index_mul = wlist_mul[i].index(j)
                    select_token_mul = masked_rep[i,token_index_mul,:]
                    select_token_label = contrastive_labels[i,j]
                    word_en_rep = torch.cat((word_en_rep,select_token_en.unsqueeze(0)),0)
                    word_mul_rep = torch.cat((word_mul_rep,select_token_mul.unsqueeze(0)),0)
                    word_labels = torch.cat((word_labels,select_token_label.unsqueeze(0)),0)
            if word_labels.sum()==0:
                num_seq -= 1
                if i==0:
                    flag = 1
                continue
            contrastive_score = torch.matmul(word_en_rep,word_mul_rep.transpose(0,1))
            logprobs = F.log_softmax(contrastive_score,dim=-1)
            gold = torch.arange(num_word).cuda(contrastive_score.get_device())
            loss = -logprobs.gather(dim=-1,index=gold.unsqueeze(1)).squeeze(1)
            loss = loss * word_labels
            loss = torch.sum(loss) / word_labels.sum()
            if i==0 or flag==1:
                sum_loss = loss
            else:
                sum_loss += loss
        avg_loss = sum_loss / num_seq
        return avg_loss    

    def compute_corr_loss(self,f_t,f_s):
        bsz, _ = f_t.size()
        f_t = F.normalize(f_t,dim=1)
        f_s = F.normalize(f_s,dim=1)
        t_simi = torch.matmul(f_t,f_t.transpose(0,1))
        s_simi = torch.matmul(f_s,f_s.transpose(0,1))
        s_exp_logits = torch.exp(s_simi)
        t_exp_logits = torch.exp(t_simi)
        s_simi_log = s_simi - torch.log(s_exp_logits.sum(1,keepdim=True))
        t_simi_log = torch.exp(t_simi)/t_exp_logits.sum(1,keepdim=True)
        corr_loss = F.kl_div(s_simi_log, t_simi_log, reduction='batchmean')
        return corr_loss

    def forward(self, input_ids1, input_ids2, token_type_ids1, token_type_ids2, attention_mask1, attention_mask2, truth_en, inp_en, seg_en, attn_msk_en, contrastive_labels, wlist_en, wlist_mul, input_ids_tlm, token_type_ids_tlm, attention_mask_tlm, labels_tlm, seg_mul, attn_msk_mul):
        
        # compute features 
        q1 = self.predictor(self.base_projector(torch.squeeze(self.bert(input_ids=input_ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)[0][:,:1,:],1)))
        
        with torch.no_grad():  # no gradient
            # compute teacher features
            k1 = self.momentum_projector(torch.squeeze(self.momentum_encoder(input_ids=input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)[0][:,:1,:],1))
        
        cls_loss = self.byol_loss(q1,k1).mean()
        # print(cls_loss)
        
        truth_rep = self.compute_teacher_representations(input_ids=truth_en, token_type_ids=seg_en, attention_mask=attn_msk_en)
        masked_TaCLsen_rep = self.compute_representations(input_ids=inp_en, token_type_ids=seg_mul, attention_mask=attn_msk_mul)
        
        tacl_loss = self.compute_tacl_loss(truth_rep, masked_TaCLsen_rep, wlist_en, wlist_mul, contrastive_labels)        
        # print('TACL_LOSS:',tacl_loss)
        
        tlm_loss = self.base_encoder(input_ids=input_ids_tlm, token_type_ids=token_type_ids_tlm, attention_mask=attention_mask_tlm, labels=labels_tlm).loss     
        
        # using different head for correlation
        q2 = self.predictor2(self.base_projector2(torch.squeeze(self.bert(input_ids=input_ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)[0][:,:1,:],1)))
         
        with torch.no_grad():  # no gradient
            k2 = self.momentum_projector2(torch.squeeze(self.momentum_encoder(input_ids=input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)[0][:,:1,:],1))

        corr_loss = self.compute_corr_loss(k2,q2)
        
        loss = cls_loss + tlm_loss + tacl_loss + 10*corr_loss

        return loss

class MMKD_mBERT(MMKD):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = 768 # encoder output dim
        
        # projectors
        self.base_projector = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_projector = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        # for param in self.momentum_projector.parameters():
        #     param.requires_grad = False  # not update by gradient
        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
        # for new head
        # projectors
        self.base_projector2 = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_projector2 = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        # for param in self.momentum_projector2.parameters():
        #     param.requires_grad = False  # not update by gradient
        # predictor 
        self.predictor2 = self._build_mlp(2, dim, mlp_dim, dim, False)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
