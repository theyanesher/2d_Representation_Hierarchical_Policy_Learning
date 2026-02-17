import clip

class LanguageConditioning(nn.Module):
    def __init__(self, pe_fix = False, self_cross_ver = 1, lang_max_seq_len = 77, lang_emb_dim = 512, num_pe_token = ,  ):
        self.pe_fix = pe_fix
        self.self_cross_ver = self_cross_ver
        self.input_dim_before_seq = 
        self.im_channels = 
        self.lang_preprocess = DenseBlock(
                lang_emb_dim,
                self.im_channels * 2,
                norm="group",
                activation=activation,
            )
        self.lang_max_seq_len = lang_max_seq_len
        self.lang_emb_dim = lang_emb_dim
        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )
        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            attn_dim,
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    def forward(self, vis_proprio_cond_data, lang_emb):
        # if self.add_lang:
        # import pdb; pdb.set_trace();
        # lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
        # _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
        # lang_goal_embs = lang_goal_embs.float()
        # else:
        #     lang_goal_embs = (
        #         torch.zeros(observation["lang_goal_embs"].shape)
        #         .float()
        #         .to(self._device)
        #     )
        import pdb; pdb.set_trace();
        language = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
        language = language.view(bs, self.lang_max_seq_len, -1)
        num_lang_tok = language.shape[1]
        import pdb; pdb.set_trace()
        vis_proprio_cond_data = torch.cat((language, vis_proprio_cond_data), dim=1)  # [B, num_img * np * np + 77, 128]
        import pdb; pdb.set_trace()
        # add learable pos encoding
        if not self.pe_fix:
            vis_proprio_cond_data = vis_proprio_cond_data + self.pos_encoding

        x = self.fc_bef_attn(vis_proprio_cond_data)
        if self.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.self_cross_ver == 1:
            import pdb; pdb.set_trace();
            lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

            # within image self attention
            imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                imgx = self_attn(imgx) + imgx
                imgx = self_ff(imgx) + imgx
            import pdb; pdb.set_trace();
            imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)
            x = torch.cat((lx, imgx), dim=1)
            # cross attention
            for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
                x = self_attn(x) + x
                x = self_ff(x) + x

        else:
            assert False

        # append language features as sequence
        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)
        