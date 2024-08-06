#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer 
from typing import Optional, List 

class CALRA(nn.Linear, LoRALayer):
    # Calra
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        N: int = 1,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.N = N
        self.fan_in_fan_out = fan_in_fan_out
        self.U, self.S, self.V = torch.linalg.svd(self.weight.data)
        #self.S = self.S/self.S.max()

        # Actual trainable parameters
        if r > 0:
            #self.lora_A = nn.Parameter(
            #    self.weight.new_zeros((r, r))
            #)
            self.lora_Es = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros(r, 1)
            ) for i in range(self.N)])

            self.lora_A = nn.Parameter(self.U[:r, :])
            self.V = nn.Parameter(self.V[:, :r])
            self.S = nn.Parameter(self.S[:r].unsqueeze(1))

            self.lora_Bs = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros((r, r))
            ) for i in range(self.N)])

            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            #self.resweight = nn.Parameter(
            #    self.weight.new_ones(1), requires_grad=True
            #)
            self.resweight = 1
            
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
            self.S.requires_grad = False
            self.U.requires_grad = False
            self.V.requires_grad = False

        self.reset_parameters()
        
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_Es'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            for i in range(self.N):
                nn.init.zeros_(self.lora_Es[i])
            #nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        if hasattr(self, 'lora_Bs'):
            for i in range(self.N):
                nn.init.normal_(self.lora_Bs[i], mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            #print ("result", result.shape)
            if self.r > 0:
                if self.N > 0:
                    E = (self.lora_Es[0] * self.S) * self.lora_Bs[0]
                if self.N > 1:
                    for i in range(1,self.N):
                        E += (self.lora_Es[i] * self.S) * self.lora_Bs[i]  #(self.lora_E + self.S)

                result2 =  self.resweight*(
                    ((self.lora_dropout(x) @ self.V) @ E) @ self.lora_A
                ) * self.scaling / (self.ranknum+1e-5)
                result += result2

            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)



class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        max_r : int = 20,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.max_r = max_r
        self.lora_dropout = lora_dropout
        # Actual trainable parameters
        self.lora_A = nn.Parameter(
            self.weight.new_zeros((max_r, in_features))
        )
        self.lora_E = nn.Parameter(
            self.weight.new_zeros(max_r, 1)
        ) 
        self.lora_B = nn.Parameter(
            self.weight.new_zeros((out_features, max_r))
        )
        self.ranknum = nn.Parameter(
            self.weight.new_zeros(1), requires_grad=False
        )
        self.ranknum.data.fill_(float(self.r))
        # self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r) 
        self.scaling = 1  
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        
        self.lora_mask = nn.Parameter(
            torch.zeros(max_r, dtype=torch.bool), requires_grad=False
        )
        self.lora_mask[:self.r] = True

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            # nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_E, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A*self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (
                    self.lora_dropout(x) @ (self.lora_A * (self.lora_mask.to(torch.uint8).unsqueeze(1)*self.lora_E)).T @ self.lora_B.T
                ) * self.scaling / (self.ranknum+1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)




class RankAllocator(object):
    """
    To be used with SVDLinear method 

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self, model, 
        lora_r:int,
        target_rank:int, 
        lora_type: str,
        init_warmup:int, 
        final_warmup:int,
        mask_interval:int,
        beta1:float, 
        beta2:float, 
        max_r : int = 20,
        total_step:Optional[int]=None, 
        target_total_rank:Optional[int]=None,
        tb_writter=None,
        tb_writter_loginterval:int=500, 
    ):
        self.ave_target_rank = target_rank 
        self.lora_type = lora_type
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r 
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step
        self.max_r = max_r
        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()
        self.total_trainable_params = 500000
        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        assert self.total_step>self.initial_warmup+self.final_warmup


    def get_lora_param_name(self):
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set))
        # if self.target_rank is None:
        #     self.target_rank = self.ave_target_rank * len(self.name_set) 

    def schedule_threshold(self, step:int):
        mask_ind = False 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        if step <= initial_warmup: 
            mask_ind = False 
        elif step > total_step - final_warmup: 
            mask_ind = True 
        else: 
            mask_ind = True if step % self.mask_interval == 0 else False 
        return None, mask_ind 

    def calculate_score(self, n):
        name_mask = n.replace("lora_E", "lora_mask")
        lora_mask = dict(self.model.named_parameters())[name_mask]
        lora_E = dict(self.model.named_parameters())[n]
        ipt_score = torch.abs(lora_E[lora_mask]).sum()
        out_features = self.shape_dict[n.replace("lora_E", "lora_B")][0] 
        in_features = self.shape_dict[n.replace("lora_E", "lora_A")][1] 
        return ipt_score/(out_features+in_features) 

    def mask_to_target_rank(self, model): 
        singular_dict = {}
        for n,p in model.named_parameters(): 
            if "lora_E" in n:
                ipt_score = self.calculate_score(n)                
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score
        
        all_is = []
        for name_mat,ipt_E in singular_dict.items():
            name_E = name_mat%"lora_E"
            all_is.append(ipt_E)
            
        all_is = torch.tensor(all_is)
        total_score = all_is.sum(dim=0)
        with torch.no_grad():
            for name_mat,ipt_E in singular_dict.items():
                name_E = name_mat%"lora_E"
                out_features = self.shape_dict[name_E.replace("lora_E", "lora_B")][0] 
                in_features = self.shape_dict[name_E.replace("lora_E", "lora_A")][1]
                params = in_features+out_features
                new_rank = int(((ipt_E*self.total_trainable_params)//(total_score*params)).item())
                if new_rank > self.max_r:
                    new_rank = self.max_r
                name_mask = name_mat%"lora_mask"
                lora_mask = dict(model.named_parameters())[name_mask]
                old_rank = torch.count_nonzero(lora_mask)
                # if (old_rank != new_rank):
                #     print(name_mat, old_rank, '->', new_rank)
                lora_E = dict(model.named_parameters())[name_E]
                lora_mask.data[:] = False
                # import ipdb; ipdb.set_trace()
                if (new_rank == self.max_r):
                    lora_mask.data[:] = True
                else:
                    imp_scores = torch.abs(lora_E).squeeze().detach()
                    thresh = torch.kthvalue(imp_scores, (self.max_r-new_rank))[0].item()
                    lora_mask.data[imp_scores > thresh] = True
                self.rank_pattern[name_E] = new_rank
            # print("")
    def update_and_mask(self, model, global_step):
        _, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            self.mask_to_target_rank(model) 
        self._maybe_tb_writter_log(model)

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step%self.log_interval==0:
            with torch.no_grad():
                regu_loss = []
                for n,p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat 
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov-I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s"%n, orth_regu.item(), self.global_step
                        )
                if len(regu_loss) > 0:
                    self.tb_writter.add_scalar(
                        "train/orth_regu_loss", sum(regu_loss)/len(regu_loss), self.global_step
                    )


def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`. 
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param

