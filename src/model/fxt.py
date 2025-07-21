import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from src.model.shortening import downsample, upsample
from src.utils.utils import compute_mean_with_padding


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm, activation_function):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if activation_function == "relu":
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == "gelu":
            activation_fn = torch.nn.GELU()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
    ):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        del activation_function

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(self.d_model, 3 * n_head * d_head)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        zero_pad = torch.zeros(
            (x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.pre_lnorm:
            w_head_q, w_head_k, w_head_v = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)

        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(
            rlen, self.n_head, self.d_head
        )  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum(
            "ibnd,jbnd->bnij", rw_head_q, w_head_k
        )  # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum(
            "ibnd,jnd->bnij", rr_head_q, r_head_k
        )  # bsz x n_head x qlen x klen
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float("inf"))
        else:
            raise NotImplementedError

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        pre_lnorm,
        activation_function,
    ):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm,
            activation_function,
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output


class BoundaryPredictor(nn.Module):
    def __init__(
        self,
        d_model,
        d_inner,
        activation_function,
        temp,
        use_binomial,
        s_lower_bound,
        bp_type,
        threshold=0.5,
    ):
        super().__init__()
        self.temp = temp
        self.use_binomial = use_binomial
        self.s_lower_bound = s_lower_bound
        self.bp_type = bp_type
        self.threshold = threshold

        if activation_function == "relu":
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == "gelu":
            activation_fn = torch.nn.GELU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, hidden, prior):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]

        self.priors = prior
        self.pred_prior = torch.tensor(
            [p[0] for p in self.priors], device=hidden.device
        )
        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)
        if self.bp_type == "gumbel":
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=boundary_probs,
            )
            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (
                hard_boundaries - soft_boundaries.detach() + soft_boundaries
            )
        elif self.bp_type in ["entropy", "unigram"]:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        self.soft_boundaries = soft_boundaries
        self.hard_boundaries = hard_boundaries

        return soft_boundaries, hard_boundaries

    def calc_loss_without_padding(self, preds, gt, attention_mask=None):
        """ """
        # B x T
        if self.bp_type in ["entropy", "unigram"]:
            assert preds is not None and gt is not None
            return self.loss(preds, gt.float())

        elif self.bp_type in ["gumbel"]:
            if attention_mask is not None and gt is None:

                # create a mask based on attention_mask
                mask = attention_mask.eq(
                    1
                )  # Mask is True where tokens are present, False for padding

                # apply the mask to predictions
                masked_preds = preds * mask.float()
                sum_preds = masked_preds.sum(dim=-1).unsqueeze(dim=-1)

                # Compute the total count of trials for each example in the batch
                total_count = mask.sum(
                    dim=-1, keepdim=True
                ).float()  # Number of non-padded tokens

            else:
                total_count = preds.size(-1)
                sum_preds = preds.sum(dim=-1)
            if self.use_binomial:
                binomial = torch.distributions.binomial.Binomial(
                    total_count, probs=self.pred_prior.to(preds.device)
                )
                loss_boundaries = -binomial.log_prob(sum_preds).mean() / preds.size(-1)

            else:
                # non-binomial loss

                est_prior = sum_preds / total_count
                prior_std = torch.tensor(
                    [p[1] for p in self.priors], device=self.pred_prior.device
                )

                upper_bound = self.pred_prior
                lower_bound = self.pred_prior - self.s_lower_bound * prior_std

                loss_high = torch.clamp(est_prior - upper_bound, min=0.0)
                loss_low = torch.clamp(lower_bound - est_prior, min=0.0)
                loss_boundaries = (loss_high + loss_low).mean()

            return loss_boundaries, self.pred_prior

    def calc_stats(self, preds, gt):
        # B x T
        preds, gt = preds.bool(), gt.bool()
        TP = ((preds == gt) & preds).sum().item()
        FP = ((preds != gt) & preds).sum().item()
        FN = ((preds != gt) & (~preds)).sum().item()

        acc = (preds == gt).sum().item() / gt.numel()

        if TP == 0:
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

        stats = {"acc": acc, "precision": precision, "recall": recall}

        return stats


class FxTTransformerLM(nn.Module):
    def __init__(
        self,
        n_token,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        pre_lnorm,
        model_config,
        activation_function,
        boundaries_type,
        spikes_left,
        temp,
        all_script_ids_dict,
        id_to_script,
        num_predictors,
        use_binomial,
        seq_len,
        s_lower_bound=3,
    ):
        super(FxTTransformerLM, self).__init__()
        self.n_token = n_token
        self.seq_len = seq_len
        self.id_to_script = id_to_script
        self.script_to_prior = all_script_ids_dict
        self.num_predictors = num_predictors
        self.use_binomial = use_binomial
        self.s_lower_bound = s_lower_bound

        # when loading the pretrained config, the keys become strings instead of int, 
        # so we convert to int here
        are_all_script_keys_string = all(
            isinstance(value, str) for value in self.id_to_script.keys()
        )
        if are_all_script_keys_string:
            self.id_to_script = {
                int(key): value
                for key, value in self.id_to_script.items()
                if key.isdigit()
            }

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = nn.Embedding(n_token, d_model)
        self.drop = nn.Dropout(dropout)

        # Relative attention specific parameters
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())

        assert pre_lnorm is False, "We didn't use pre_lnorm"

        def create_decoder_layers(n_layers):
            layers = nn.ModuleList(
                [
                    RelPartialLearnableDecoderLayer(
                        n_head,
                        d_model,
                        d_head,
                        d_inner,
                        dropout,
                        dropatt=dropatt,
                        pre_lnorm=pre_lnorm,
                        activation_function=activation_function,
                    )
                    for _ in range(n_layers)
                ]
            )

            return layers

        pre_layers, (shortened_layers,), post_layers = eval(model_config)

        self.boundaries_type = boundaries_type
        self.is_bp = boundaries_type in ["unigram", "entropy", "gumbel"]

        if post_layers == 0 and shortened_layers == 0:
            assert boundaries_type == "none"
            self.layers = nn.ModuleList([create_decoder_layers(pre_layers)])
        else:
            self.null_group = nn.Parameter(torch.Tensor(1, 1, d_model).zero_())
            nn.init.normal_(self.null_group)

            self.layers = nn.ModuleList(
                [
                    create_decoder_layers(pre_layers),
                    create_decoder_layers(shortened_layers),
                    create_decoder_layers(post_layers),
                ]
            )

            self.down_ln = nn.LayerNorm(d_model)

            # Create boundary predictor layers
            if self.is_bp:
                self.script_to_bp_layers = nn.ModuleDict(
                    {
                        script: BoundaryPredictor(
                            d_model=d_model,
                            d_inner=d_inner,
                            activation_function=activation_function,
                            temp=temp,
                            use_binomial=self.use_binomial,
                            s_lower_bound=self.s_lower_bound,
                            bp_type=boundaries_type,
                        )
                        for i, (script, pri) in itertools.islice(
                            enumerate(
                                zip(
                                    all_script_ids_dict.keys(),
                                    all_script_ids_dict.values(),
                                )
                            ),
                            self.num_predictors,
                        )  # itertools.islice is used to limit the number of predictors
                    }
                )

                self.spikes_left = spikes_left

        self.final_cast = nn.Linear(d_model, n_token)
        self.crit = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def _forward(self, core_input, layers):
        # Core_input is of size (T x B x C)
        qlen, _, _ = core_input.size()

        dec_attn_mask = torch.triu(core_input.new_ones(qlen, qlen), diagonal=1).bool()

        pos_seq = torch.arange(
            qlen - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype
        )
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = core_input
        for i, layer in enumerate(layers):
            core_out = layer(
                core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask
            )

        return core_out

    def get_spikes(self, vector):
        total = torch.ones_like(vector).bool()

        for i in range(1, self.spikes_left + 1, 1):
            mask = vector[i:] > vector[:-i]
            total[i:] &= mask

        return total

    def compute_compression_rate(self, hard_boundaries, attention_mask):
        # Create a mask based on attention_mask
        mask = attention_mask.eq(
            1
        )  # Mask is True where tokens are present, False for padding

        # Apply the mask to hard_boundaries
        masked_hard_boundaries = hard_boundaries * mask.float()

        # Compute the total number of non-padded positions for each row in the batch
        num_non_padded_positions_per_row = mask.sum(
            dim=1
        ).float()  # Count the number of non-padded positions for each row

        # Compute the sum of predictions only on non-padded positions for each row in the batch
        sum_hard_boundaries_non_padded_per_row = masked_hard_boundaries.sum(
            dim=1
        )  # Sum of hard_boundaries for each row

        # Compute the compression_rate only on non-padded positions for each row in the batch
        zero_mask = sum_hard_boundaries_non_padded_per_row.eq(0)
        if zero_mask.any():
            sum_hard_boundaries_non_padded_per_row[zero_mask] = 1
        compression = (
            num_non_padded_positions_per_row / sum_hard_boundaries_non_padded_per_row
        )
        compression_rate = compression.mean()
        compression_variance = compression.var()
        # change the NaN values in compression_variance to 0
        compression_variance = torch.nan_to_num(compression_variance, nan=0.0)
        compression_std = compression.std()

        if torch.isnan(compression_rate):
            print("Debug Info:")
            print("num_non_padded_positions_per_row:", num_non_padded_positions_per_row)
            print(
                "sum_hard_boundaries_non_padded_per_row:",
                sum_hard_boundaries_non_padded_per_row,
            )

        p_ones = (
            sum_hard_boundaries_non_padded_per_row / num_non_padded_positions_per_row
        ).mean()

        return (compression_rate, compression_variance, compression_std), p_ones

    def compute_boundaries_in_parallel(
        self, hidden, target, dtype, boundary_predictor, priors, device
    ):
        embeddings = hidden.clone()
        residual = None
        pre_upsample = None
        shortened_length = None
        soft_boundaries = None
        hard_boundaries = None
        # eot_emb = self.word_emb(
        #     torch.tensor([[self.n_token-1]],  device=device)
        # )
        # Process input with Transformer blocks
        for i in range(len(self.layers)):
            if i == 1:  # Downsampling
                # residual = hidden
                residual = hidden.clone()
                bp_input = hidden  # + embeddings
                soft_boundaries, hard_boundaries = boundary_predictor(
                    bp_input, prior=priors
                )

                # B x T
                hidden = downsample(
                    boundaries=hard_boundaries,
                    hidden=hidden,
                    null_group=self.null_group,
                )
                hidden = self.down_ln(hidden)

                # Shortening stats

                # Shortened length might not really reflect true length with padding
                shortened_length = hidden.size(
                    0
                )  # no longer useful since it has mulitple scripts in it.

            elif i == 2:  # Upsampling
                pre_upsample = hidden

                back_hidden = upsample(
                    boundaries=hard_boundaries,
                    shortened_hidden=hidden,
                )

                hidden = back_hidden + residual

            # Out of downsample / upsample -> regular Transformer blocks
            layers = self.layers[i]
            self.last_hidden = hidden

            hidden = self._forward(core_input=hidden, layers=layers)
        return (
            hidden,
            pre_upsample,
            target,
            shortened_length,
            soft_boundaries,
            hard_boundaries,
        )

    def forward(self, batch, task):
        """
        Data: Batch Size x Sequence length  --> Sequence length x Batch Size
        Attention_mask: Batch Size x Sequence length  --> Batch Size x Sequence length
        """
        self.task = task
        data = batch["input_ids"]
        # note that batch['attention_mask'] has been taken down since I only need it during finetuning

        # In each batch, get all the unique script ids and check that they are contained in script ids
        unique_script_ids = torch.unique(data[:, 0])
        batch_script_ids = data[:, 0]
        batch_scripts = [
            self.id_to_script[script_id] for script_id in batch_script_ids.tolist()
        ]
        batch_priors = [self.script_to_prior[script_id] for script_id in batch_scripts]
        assert all(
            value in self.id_to_script.keys() for value in unique_script_ids.tolist()
        )
        batch_dict = {}
        overall_stats = {}

        batch_dict["input_ids"] = data[:, 1:].T
        # We shift the input ids by 1 when computing the loss
        target_ids = batch_dict["input_ids"].clone()
        tgt_len = target_ids.size(0)
        embeddings = self.drop(self.word_emb(batch_dict["input_ids"]))
        # (Tokenization happens here) Downsample and upsample representations
        if self.is_bp:
            available_bp_id = list(self.script_to_bp_layers.keys())[0]
            boundary_predictor = self.script_to_bp_layers[available_bp_id]
        else:
            boundary_predictor = None
        (
            hidden,
            pre_upsample,
            target_ids,
            shortened_length,
            soft_boundaries,
            hard_boundaries,
        ) = self.compute_boundaries_in_parallel(
            embeddings,
            target_ids,
            dtype=data.dtype,
            boundary_predictor=boundary_predictor,
            priors=batch_priors,
            device=data.device,
        )
        loss_boundaries = torch.tensor(0.0, device=data.device)
        if self.is_bp:
            # Calculate boundary loss here
            soft_boundaries = soft_boundaries[:, -tgt_len:]
            hard_boundaries = hard_boundaries[:, -tgt_len:]
            if task == "LM":
                loss_boundaries, pred_priors = self.script_to_bp_layers[
                    available_bp_id
                ].calc_loss_without_padding(
                    preds=hard_boundaries, gt=None, attention_mask=None
                )
            else:
                # check the shape of the attention mask
                loss_boundaries, pred_priors = self.script_to_bp_layers[
                    available_bp_id
                ].calc_loss_without_padding(
                    preds=hard_boundaries,
                    gt=None,
                    attention_mask=batch["attention_mask"],
                )
            # group pred_priors by script id add to stats

            if task == "generation":
                logit = self.final_cast(hidden)
                return logit, pre_upsample, self.last_hidden
            elif task == "tokenization2":
                overall_stats["hard_boundaries"] = (
                    hard_boundaries  # if I want to the boundaries
                )
                overall_stats["priors"] = pred_priors.tolist()

                return None, overall_stats, None
            for script_id in unique_script_ids.tolist():
                mask = batch_script_ids == script_id  # Mask for current script ID
                script_pred_prior = pred_priors[mask].mean()  # Calculate mean

                # script compression rate
                script_hard_boundaries = hard_boundaries[mask]
                script_attention_mask = batch["attention_mask"][mask]
                script_compression_rate, script_p_ones = self.compute_compression_rate(
                    script_hard_boundaries, script_attention_mask
                )

                overall_stats[f"{self.id_to_script[script_id]}_prior"] = (
                    script_pred_prior.item()
                )
                overall_stats[f"{self.id_to_script[script_id]}_compression_rate"] = (
                    script_compression_rate[0].item()
                )
                overall_stats[f"{self.id_to_script[script_id]}_compression_var"] = (
                    script_compression_rate[1].item()
                )
                overall_stats[f"{self.id_to_script[script_id]}_p_ones"] = (
                    script_p_ones.item()
                )
            overall_stats["shortened_length"] = shortened_length
            overall_stats["bp_loss"] = loss_boundaries.item()

        # Get logits
        logit = self.final_cast(hidden)
        shift_logits = logit[:-1].contiguous()
        shift_labels = target_ids[1:].contiguous()
        loss_boundaries = loss_boundaries.reshape(-1)
        loss = self.crit(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        if task == "LM":
            return loss, overall_stats, loss_boundaries, shift_logits

        else:
            if task == "tokenization2":
                overall_stats["hard_boundaries"] = (
                    hard_boundaries  # if I want to the boundaries
                )
                overall_stats["priors"] = pred_priors.tolist()
            return hidden, overall_stats, loss_boundaries


class FxTAverageSingleInputWithPadding(nn.Module):
    """
    Sequence classification over Single Inputs sequences.
    We take the average of token-level representations without including padded tokens

    """

    # compute loss over non-padded tokens
    def __init__(self, num_labels, pretrained_mem_transformer, task="seq_cls"):
        super(FxTAverageSingleInputWithPadding, self).__init__()
        self.memtransformer = pretrained_mem_transformer
        self.score = nn.Linear(
            pretrained_mem_transformer.d_model, num_labels, bias=False
        )
        self.num_labels = num_labels
        self.fct = nn.CrossEntropyLoss()
        self.task = task

    def forward(self, input_batch):
        # get the number of in
        # hidden_states, stats, boundary_loss = self.memtransformer(input_batch["input_ids"], input_batch["input_ids"].clone(), task="class")
        hidden_states, stats, boundary_loss = self.memtransformer(
            input_batch, task=self.task
        )
        # Compute mean without considering padding
        hidden_states = hidden_states.permute(1, 0, 2)
        if self.task == "seq_cls":
            hidden_states = compute_mean_with_padding(
                hidden_states, input_batch["attention_mask"]
            )
        if self.task == "token_cls":
            # Slicing creates a non-contiguous tensor that can't be viewed directly
            # Use .contiguous() to create a contiguous copy before using view()
            input_batch["labels"] = input_batch["labels"][
                :, :-1
            ].contiguous()  # removing the last token that was induced due to lang id
        # hidden_states = torch.mean(hidden_states, dim=0)
        logits = self.score(hidden_states)

        loss = self.fct(
            logits.view(-1, self.num_labels), input_batch["labels"].view(-1)
        )

        return loss, logits, stats, boundary_loss
