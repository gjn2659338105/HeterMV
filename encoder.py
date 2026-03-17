import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import RGCNConv


class HeteroGraphReasoning(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, relation_types, alpha=0.1, dropout=0.1):
        super().__init__()
        self.relation2id = {rel: i for i, rel in enumerate(relation_types)}
        self.layers = nn.ModuleList([
            RGCNConv(in_dim if i == 0 else out_dim, out_dim, len(relation_types))
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.alpha = alpha  # Controls the strength of neighbor aggregation

    def forward(self, x, edge_index_dict):
        """
        Args:
            x: Node features of shape [num_nodes, hidden_size]
            edge_index_dict: Dict[str, Tensor], each tensor has shape [2, num_edges]
        Returns:
            Updated node representations with residual graph refinement
        """
        device = x.device
        edge_index_all = []
        edge_type_all = []

        for rel, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            edge_index = edge_index.to(device)
            rel_id = self.relation2id[rel]
            edge_index_all.append(edge_index)
            edge_type_all.append(
                torch.full(
                    (edge_index.size(1),),
                    rel_id,
                    dtype=torch.long,
                    device=device
                )
            )

        # If no edge exists in this view, directly return x
        if len(edge_index_all) == 0:
            return x

        edge_index = torch.cat(edge_index_all, dim=1)
        edge_type = torch.cat(edge_type_all, dim=0)

        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            h = layer(x, edge_index, edge_type)
            h = dropout(norm(h))
            x = x + self.alpha * h  # Residual fusion with center-node dominance

        return x


class PromptLearner(nn.Module):
    def __init__(self, config, args, data, token_embs):
        super().__init__()
        self.current_device = args.device
        self.language_model = args.language_model
        self.num_prompt_embs = args.num_prompt_embs
        self.random_prompt_init = args.random_prompt_init
        self.hidden_size = args.hidden_size
        self.num_labels = args.num_labels
        self.training_claim_ids = data.training_claim_ids
        self.training_labels = data.training_labels
        self.token_embs = token_embs
        self.num_sampled_references = args.num_sampled_references
        self.pretrained_model_name_or_path = args.pretrained_model_name_or_path

        self.linear_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size)
        ])
        self.tanh = nn.Tanh()
        self.temperature = 100.0

        self.init_prompts(data)

        # One learnable type embedding for each label
        self.prompt_type_emb = nn.Embedding(self.num_labels, self.hidden_size)

    def init_prompts(self, data):
        """
        Initialize label-specific prompt embeddings.
        If random_prompt_init is True, prompts are sampled from the vocabulary embedding table.
        Otherwise, prompts are initialized by averaging claim/evidence token embeddings.
        """
        if self.random_prompt_init:
            vocab_size = self.token_embs.num_embeddings
            token_ids = torch.randint(
                low=0,
                high=vocab_size,
                size=(self.num_labels, self.num_prompt_embs),
                dtype=torch.long
            )
            prompt_embs = self.token_embs.weight[token_ids]  # [num_labels, prompt_len, hidden]
        else:
            prompt_embs_list = []

            for label_id in range(self.num_labels):
                claim_ids_one_label = data.label_id2training_claim_ids[label_id]

                if len(claim_ids_one_label) == 0:
                    # Fallback to random init if no sample exists for this label
                    rand_ids = torch.randint(
                        low=0,
                        high=self.token_embs.num_embeddings,
                        size=(1, self.num_prompt_embs),
                        dtype=torch.long
                    )
                    prompt_embs_list.append(self.token_embs.weight[rand_ids])
                    continue

                claim_input_ids_one_label = np.array([
                    data.claim_input_ids[claim_id][1:self.num_prompt_embs + 1]
                    for claim_id in claim_ids_one_label
                ])
                claim_input_ids_one_label = torch.LongTensor(claim_input_ids_one_label)

                evid_ids_one_label = np.array([
                    data.sampled_evid_ids[claim_id] for claim_id in claim_ids_one_label
                ])
                evid_ids_one_label = np.reshape(evid_ids_one_label, [-1])

                evid_input_ids_one_label = np.array([
                    data.evid_input_ids[evid_id][1:self.num_prompt_embs + 1]
                    for evid_id in evid_ids_one_label
                ])
                evid_input_ids_one_label = torch.LongTensor(evid_input_ids_one_label)

                with torch.no_grad():
                    prompt_embs_claim = self.token_embs(claim_input_ids_one_label)
                    prompt_embs_evid = self.token_embs(evid_input_ids_one_label)
                    prompt_embs = torch.cat([prompt_embs_claim, prompt_embs_evid], dim=0)
                    prompt_embs = prompt_embs.mean(dim=0, keepdim=True)  # [1, prompt_len, hidden]

                prompt_embs_list.append(prompt_embs)

            prompt_embs = torch.cat(prompt_embs_list, dim=0)

        self.prompt_embs = nn.Parameter(prompt_embs)

    def forward(self, input_ids, token_embs, evid_emb):
        """
        Build label-aware prompted inputs.
        Args:
            input_ids: [B, L]
            token_embs: embedding layer
            evid_emb: [B, D], fused evidence representation
        Returns:
            prompts_list: a list of length num_labels,
                          each element is [B, L + prompt_len + 1, D]
        """
        inputs_embeds = token_embs(input_ids)  # [B, L, D]
        B = inputs_embeds.size(0)

        scaling = self.tanh(self.linear_layers[0](evid_emb) / self.temperature)   # [B, D]
        shifting = self.tanh(self.linear_layers[1](evid_emb) / self.temperature)  # [B, D]

        scaling = scaling.unsqueeze(1).expand(-1, self.num_prompt_embs + 1, -1)   # [B, P+1, D]
        shifting = shifting.unsqueeze(1).expand(-1, self.num_prompt_embs + 1, -1) # [B, P+1, D]

        prompts_list = []

        for label_id in range(self.num_labels):
            # Label-specific prompt tokens: [P, D] -> [B, P, D]
            prompt_embs = self.prompt_embs[label_id].unsqueeze(0).expand(B, -1, -1)

            # Label type token: [D] -> [B, 1, D]
            type_emb = self.prompt_type_emb.weight[label_id].unsqueeze(0).unsqueeze(1).expand(B, 1, -1)

            # Full prompt = [type token] + [prompt tokens]
            full_prompt = torch.cat([type_emb, prompt_embs], dim=1)  # [B, P+1, D]

            # Evidence-conditioned prompt adaptation
            adjusted_prompt = full_prompt * (1 + scaling) + shifting

            # Concatenate as: [CLS] + prompt + remaining tokens
            prompted_inputs = torch.cat(
                [inputs_embeds[:, :1, :], adjusted_prompt, inputs_embeds[:, 1:, :]],
                dim=1
            )
            prompts_list.append(prompted_inputs)

        return prompts_list


class EncoderLayer(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.language_model = args.language_model
        self.num_sampled_evidence = args.num_sampled_evidence
        self.num_sampled_references = args.num_sampled_references
        self.hidden_size = args.hidden_size
        self.num_hidden_layers = args.num_hidden_layers
        self.has_contexts = args.has_contexts
        self.has_references = args.has_references
        self.num_prompt_embs = args.num_prompt_embs

        # Independent GNN for each view
        self.graph_reasoning_dict = nn.ModuleDict({
            view: HeteroGraphReasoning(
                in_dim=self.hidden_size,
                out_dim=self.hidden_size,
                num_layers=2,
                relation_types=["evid-evid", "ctx-evid", "ref-evid"]
            )
            for view in ["evidence-only", "evid-ctx", "evid-ref", "fusion"]
        })

    def build_edges(self, num_evid, view, ctx_offset, ref_offset):
        """
        Build graph edges for each view.

        Node indexing:
            0 ~ ctx_offset - 1              : evidence nodes
            ctx_offset ~ ref_offset - 1     : context nodes
            ref_offset ~ end                : reference nodes
        """
        edge_index_dict = {
            "evid-evid": [],
            "ctx-evid": [],
            "ref-evid": []
        }

        sample_num_evid = self.num_sampled_evidence
        num_ref_per_evid = self.num_sampled_references

        num_groups = num_evid // sample_num_evid

        for g in range(num_groups):
            group_start = g * sample_num_evid
            group_end = group_start + sample_num_evid

            if view in ["evidence-only", "fusion"]:
                # Other evidence nodes in the same claim aggregate into each target evidence
                for i in range(group_start, group_end):
                    for j in range(group_start, group_end):
                        if j != i:
                            edge_index_dict["evid-evid"].append((j, i))

            if view in ["evid-ctx", "fusion"]:
                # Each context node connects to its corresponding evidence node
                for i in range(group_start, group_end):
                    ctx_id = ctx_offset + i
                    edge_index_dict["ctx-evid"].append((ctx_id, i))

            if view in ["evid-ref", "fusion"]:
                # All sampled references connect to their corresponding evidence node
                for i in range(group_start, group_end):
                    for r in range(num_ref_per_evid):
                        ref_id = ref_offset + i * num_ref_per_evid + r
                        edge_index_dict["ref-evid"].append((ref_id, i))

        # Convert to tensor
        for key in edge_index_dict:
            if len(edge_index_dict[key]) > 0:
                edge_index_dict[key] = torch.tensor(edge_index_dict[key], dtype=torch.long).t().contiguous()
            else:
                edge_index_dict[key] = torch.empty((2, 0), dtype=torch.long)

        return edge_index_dict

    def forward(
        self,
        lm,
        hidden_states,
        attention_mask,
        ctx_hidden_states=None,
        ctx_attention_mask=None,
        ref_hidden_states=None,
        ref_attention_mask=None,
        claim_or_evid='evid',
        mode='train'
    ):
        """
        Encode text with the transformer backbone first,
        then inject graph-based multi-view reasoning for evidence.
        """
        for layer_id in range(self.num_hidden_layers):
            hidden_states = lm.encoder.layer[layer_id](hidden_states, attention_mask=attention_mask)[0]

            if ctx_hidden_states is not None:
                ctx_hidden_states = lm.encoder.layer[layer_id](
                    ctx_hidden_states, attention_mask=ctx_attention_mask
                )[0]

            if ref_hidden_states is not None:
                ref_hidden_states = lm.encoder.layer[layer_id](
                    ref_hidden_states, attention_mask=ref_attention_mask
                )[0]

        if claim_or_evid == 'claim':
            return hidden_states

        # Position 3 corresponds to the original [CLS] after prepending 3 slots
        evid_cls = hidden_states[:, 3, :]
        ctx_cls = ctx_hidden_states[:, 3, :] if ctx_hidden_states is not None else None
        ref_cls = ref_hidden_states[:, 3, :] if ref_hidden_states is not None else None

        node_list = [evid_cls]
        ctx_offset = evid_cls.size(0)
        ref_offset = ctx_offset + (ctx_cls.size(0) if ctx_cls is not None else 0)

        if ctx_cls is not None:
            node_list.append(ctx_cls)
        if ref_cls is not None:
            node_list.append(ref_cls)

        node_reps = torch.cat(node_list, dim=0)

        view_to_slot = {
            "evidence-only": 0,
            "evid-ctx": 1,
            "evid-ref": 2,
            "fusion": 3,
        }

        for view_name, slot_id in view_to_slot.items():
            edge_index_dict = self.build_edges(
                num_evid=evid_cls.size(0),
                view=view_name,
                ctx_offset=ctx_offset,
                ref_offset=ref_offset
            )
            gnn = self.graph_reasoning_dict[view_name]
            node_reps_view = gnn(node_reps, edge_index_dict)

            # Write evidence-node outputs into dedicated structural slots
            hidden_states[:, slot_id, :] = node_reps_view[:evid_cls.size(0)]

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, lm, config, args, data):
        super().__init__()
        self.language_model = args.language_model
        self.num_sampled_evidence = args.num_sampled_evidence
        self.num_sampled_references = args.num_sampled_references
        self.hidden_size = args.hidden_size
        self.num_prompt_embs = args.num_prompt_embs
        self.num_labels = args.num_labels
        self.has_contexts = args.has_contexts
        self.has_references = args.has_references

        self.encoder_layer = EncoderLayer(config, args)
        token_embs = lm.embeddings.word_embeddings
        self.prompt_learner = PromptLearner(config, args, data, token_embs)

    def prepend_hidden_states_and_attention_mask(self, hidden_states, attention_mask):
        """
        Prepend 3 structural slots to hidden states and attention masks.
        These slots are used to store graph-view outputs later.
        """
        num_texts = hidden_states.size(0)

        slot_placeholder = torch.zeros(
            [num_texts, 3, hidden_states.size(-1)],
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        hidden_states = torch.cat([slot_placeholder, hidden_states], dim=1)

        slot_mask = torch.zeros(
            [num_texts, 3],
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        attention_mask = torch.cat([slot_mask, attention_mask], dim=-1)

        return hidden_states, attention_mask

    def encoder(
        self,
        lm,
        hidden_states,
        attention_mask,
        ctx_input_ids=None,
        ctx_attention_mask=None,
        ref_input_ids=None,
        ref_attention_mask=None,
        claim_or_evid='evid',
        mode='train'
    ):
        hidden_states, attention_mask = self.prepend_hidden_states_and_attention_mask(hidden_states, attention_mask)

        ctx_hidden_states, ctx_extended_attention_mask = None, None
        if ctx_input_ids is not None:
            ctx_hidden_states = lm.embeddings(input_ids=ctx_input_ids)
            ctx_hidden_states, ctx_attention_mask = self.prepend_hidden_states_and_attention_mask(
                ctx_hidden_states, ctx_attention_mask
            )
            ctx_extended_attention_mask = (1.0 - ctx_attention_mask[:, None, None, :]) * -10000.0

        ref_hidden_states, ref_extended_attention_mask = None, None
        if ref_input_ids is not None:
            ref_hidden_states = lm.embeddings(input_ids=ref_input_ids)
            ref_hidden_states, ref_attention_mask = self.prepend_hidden_states_and_attention_mask(
                ref_hidden_states, ref_attention_mask
            )
            ref_extended_attention_mask = (1.0 - ref_attention_mask[:, None, None, :]) * -10000.0

        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        hidden_states = self.encoder_layer(
            lm=lm,
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
            ctx_hidden_states=ctx_hidden_states,
            ctx_attention_mask=ctx_extended_attention_mask,
            ref_hidden_states=ref_hidden_states,
            ref_attention_mask=ref_extended_attention_mask,
            claim_or_evid=claim_or_evid,
            mode=mode
        )

        if claim_or_evid == 'claim':
            # Position 3 is the original [CLS]
            claim_cls = hidden_states[:, 3, :]
            return claim_cls

        B = hidden_states.size(0) // self.num_sampled_evidence

        # Structural slots: 0~3 correspond to four graph views
        evid_only_emb = hidden_states[:, 0, :]
        evid_ctx_emb = hidden_states[:, 1, :] if ctx_input_ids is not None else None
        evid_ref_emb = hidden_states[:, 2, :] if ref_input_ids is not None else None
        fusion_emb = hidden_states[:, 3, :]

        # Aggregate evidence-level representations into claim-level representations
        evid_only_emb = evid_only_emb.reshape(B, self.num_sampled_evidence, -1).mean(dim=1)
        fusion_emb = fusion_emb.reshape(B, self.num_sampled_evidence, -1).mean(dim=1)

        if evid_ctx_emb is not None:
            evid_ctx_emb = evid_ctx_emb.reshape(B, self.num_sampled_evidence, -1).mean(dim=1)

        if evid_ref_emb is not None:
            evid_ref_emb = evid_ref_emb.reshape(B, self.num_sampled_evidence, -1).mean(dim=1)

        return fusion_emb, evid_only_emb, evid_ctx_emb, evid_ref_emb

    def forward(
        self,
        lm,
        input_ids,
        attention_mask,
        ctx_input_ids=None,
        ctx_attention_mask=None,
        ref_input_ids=None,
        ref_attention_mask=None,
        evid_emb=None,
        claim_or_evid='evid',
        mode='train'
    ):
        num_texts = input_ids.size(0)

        if claim_or_evid == 'claim':
            token_embs = lm.embeddings.word_embeddings
            prompts_list = self.prompt_learner(input_ids, token_embs, evid_emb)

            prompt_mask = torch.ones(
                [num_texts, self.num_prompt_embs + 1],
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat(
                [attention_mask[:, :1], prompt_mask, attention_mask[:, 1:]],
                dim=-1
            )

            text_emb_list = []
            for label_id in range(self.num_labels):
                hidden_states = lm.embeddings(inputs_embeds=prompts_list[label_id])
                text_emb = self.encoder(
                    lm=lm,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    claim_or_evid=claim_or_evid,
                    mode=mode
                )
                text_emb_list.append(text_emb)

            return text_emb_list

        elif claim_or_evid == 'evid':
            hidden_states = lm.embeddings(input_ids=input_ids)

            fusion_emb, evid_only_emb, evid_ctx_emb, evid_ref_emb = self.encoder(
                lm=lm,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                ctx_input_ids=ctx_input_ids,
                ctx_attention_mask=ctx_attention_mask,
                ref_input_ids=ref_input_ids,
                ref_attention_mask=ref_attention_mask,
                claim_or_evid=claim_or_evid,
                mode=mode
            )

            return fusion_emb, evid_only_emb, evid_ctx_emb, evid_ref_emb

        else:
            raise ValueError(f"Unsupported claim_or_evid: {claim_or_evid}")