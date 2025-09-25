import enum
import torch
import random
import numpy as np
import torch as th
import gaussian_diffusion as gd
from diffusion import q_sample

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        steps = 1000,
        noise_schedule="linear",
        predict_xstart = False,
        learn_sigma = False,
        sigma_small = False,
        use_kl = False,
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
    ):
        #self.simplex = Simplex_CLASS()
        #self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, False)

        self.model_mean_type = (
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        self.model_var_type = (
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        )
        if use_kl:
            self.loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            self.loss_type = gd.LossType.RESCALED_MSE
        else:
            self.loss_type = gd.LossType.MSE
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = gd.get_named_beta_schedule(noise_schedule, steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        print('self.num_timesteps', self.num_timesteps)

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def p_sample_loop(
        self,
        model,
        shape,
        noise,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,

    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"] 

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x[:, :4, ...])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            a, out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        else:
            a=0*noise
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "saliency": a}

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        time=1000,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        org=None,
        model_kwargs=None,
        device=None,
        progress=False,
        classifier=None):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise   #输入加噪后的图像
        else:
            img = th.randn(*shape, device=device)
      
        indices = list(range(time))[::-1]
        print('indices', indices)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
          
        for i in indices:
                t = th.tensor([i] * shape[0], device=device)

                with th.no_grad():
                    
                    out = self.p_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                    yield out
                    img = out["sample"]

    def ddim_sample_loop(
        self,
        model,
        shape,
        timestep,
        noise,
        inputs,
        concate,
        diffusion_mu,
        vss1_forward,
        vss2_forward,
        vss3_forward,
        predict_xstart = False,
        learn_sigma = False,
        sigma_small = False,
        use_kl = False,
        rescale_learned_sigmas = False,
        rescale_timesteps = False,
        clip_denoised = False,
        denoised_fn = None,
        cond_fn = None,
        model_kwargs = None,
        device = None,
        progress = True,
        eta=0.0,
        ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        #t = th.randint(0,1000, (b,), device=device).long().to(device) #ddim前向加噪过程的时间步参数，随机采样得到

        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            timestep,
            noise,
            inputs,
            concate,
            diffusion_mu,
            vss1_forward,
            vss2_forward,
            vss3_forward,
            predict_xstart = predict_xstart,
            learn_sigma = learn_sigma,
            sigma_small = sigma_small,
            use_kl = use_kl,
            rescale_learned_sigmas = rescale_learned_sigmas,
            rescale_timesteps = rescale_timesteps,
            clip_denoised=clip_denoised,
            denoised_fn = denoised_fn,
            cond_fn = cond_fn,
            model_kwargs = model_kwargs,
            device = device,
            progress = progress,
            eta=eta,
        ):

            final = sample   #保存最后一个样本结果的字典
            #print(final["sample"].shape)
        #viz.image(visualize(final["sample"].cpu()[0, ...]), opts=dict(caption="sample"+ str(10) ))
        return final["sample"]   #final是字典，sample是键值

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        timestep,
        noise,
        inputs,
        concate,
        diffusion_mu,
        vss1_forward,
        vss2_forward,
        vss3_forward,
        predict_xstart = False,
        learn_sigma = False,
        sigma_small = False,
        use_kl = False,
        rescale_learned_sigmas = False,
        rescale_timesteps = False,
        clip_denoised = False,
        denoised_fn = None,
        cond_fn = None,
        model_kwargs = None,
        device = None,
        progress = True,
        eta=0.0,
        ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(timestep-1))[::-1]
        #print('indices', len(indices)) #输出列表

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
        for i in indices:

            k=abs(timestep-1-i)
            #if k%20==0:
            #    print('k',k)
            #生成时间特征的参数编码
            t = th.tensor([k] * shape[0], device=device)
            
            with th.no_grad():
                
                out = self.ddim_reverse_sample(
                    model,
                    img,
                    t,
                    inputs,
                    concate,  
                    diffusion_mu,
                    vss1_forward,
                    vss2_forward,
                    vss3_forward,
                    predict_xstart = predict_xstart,
                    learn_sigma = learn_sigma,
                    sigma_small = sigma_small,
                    use_kl = use_kl,
                    rescale_learned_sigmas= rescale_learned_sigmas,
                    rescale_timesteps= rescale_timesteps,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
               )

                yield out  
                img = out["sample"]
        
                
        for i in indices:
                t = th.tensor([i] * shape[0], device=device)

                with th.no_grad():
                    out = self.ddim_sample(
                       model,
                       img,
                       t,
                       inputs,
                       concate,
                       diffusion_mu,
                       vss1_forward,
                       vss2_forward,
                       vss3_forward,
                       predict_xstart = predict_xstart,
                       learn_sigma = learn_sigma,
                       sigma_small = sigma_small,
                       use_kl = use_kl,
                       rescale_learned_sigmas = rescale_learned_sigmas,
                       rescale_timesteps= rescale_timesteps,
                       clip_denoised = clip_denoised,
                       denoised_fn = denoised_fn,
                       cond_fn = cond_fn,
                       model_kwargs = model_kwargs,
                       eta = eta,
                    )
                   
                    yield out
                    img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            inputs,
            concate,
            diffusion_mu,
            vss1_forward,
            vss2_forward,
            vss3_forward,
            predict_xstart,
            learn_sigma,
            sigma_small,
            use_kl,
            rescale_learned_sigmas,
            rescale_timesteps,
            clip_denoised = False,
            denoised_fn = None,
            cond_fn = None,
            model_kwargs = None,
            eta=0.0,
        ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            inputs,
            concate,
            diffusion_mu,
            vss1_forward,
            vss2_forward,
            vss3_forward,
            predict_xstart = predict_xstart,
            learn_sigma = learn_sigma,
            sigma_small = sigma_small,
            use_kl = use_kl,
            rescale_learned_sigmas = rescale_learned_sigmas,
            rescale_timesteps = rescale_timesteps,
            clip_denoised = clip_denoised,
            denoised_fn = denoised_fn,
            model_kwargs = model_kwargs,
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"] }

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        inputs,
        concate,
        diffusion_mu,
        vss1_forward,
        vss2_forward,
        vss3_forward,
        predict_xstart,
        learn_sigma,
        sigma_small,
        use_kl,
        rescale_learned_sigmas,
        rescale_timesteps,
        clip_denoised = False,
        denoised_fn = None,
        model_kwargs = None,
        eta=0.0,
        ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            inputs,
            concate,
            diffusion_mu,
            vss1_forward,
            vss2_forward,
            vss3_forward,
            predict_xstart = predict_xstart,
            learn_sigma = learn_sigma,
            sigma_small = sigma_small,
            use_kl = use_kl,
            rescale_learned_sigmas= rescale_learned_sigmas,
            rescale_timesteps= rescale_timesteps,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance(
            self, model, x, t, inputs, concate, diffusion_mu, vss1_forward, vss2_forward, vss3_forward, predict_xstart, learn_sigma, sigma_small, use_kl, rescale_learned_sigmas, rescale_timesteps, clip_denoised=False, denoised_fn=None, model_kwargs=None
        ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        self.model_mean_type = ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X
        self.model_var_type = (
            (
                ModelVarType.FIXED_LARGE
                if not sigma_small
                else ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else ModelVarType.LEARNED_RANGE
        )
        if use_kl:
            self.loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            self.loss_type = gd.LossType.RESCALED_MSE
        else:
            self.loss_type = gd.LossType.MSE
        self.rescale_timesteps = rescale_timesteps

        B, C = x.shape[:2]
        
        #assert t.shape == (B,)
        noise1 = torch.randn_like(inputs[0])
        noise2 = torch.randn_like(inputs[1])
        noise3 = torch.randn_like(inputs[2])
        input1_noisy = q_sample(inputs[0], t, noise1)
        input2_noisy = q_sample(inputs[1], t, noise2)
        input3_noisy = q_sample(inputs[2], t, noise3)
        
        vss1_fout, vss1_fout1, vss1_fout2, vss1_fout3, vss1_fout4, original_tf1, tf1  = vss1_forward(input1_noisy, t) #[b,256,56,56]
        vss2_fout, vss2_fout1, vss2_fout2, vss2_fout3, vss2_fout4, original_tf2, tf2  = vss2_forward(input2_noisy, t) #[b,512,28,28] 
        vss3_fout, vss3_fout1, vss3_fout2, vss3_fout3, vss3_fout4, original_tf3, tf3  = vss3_forward(input3_noisy, t) #[b,1024,14,14]
        
        concatenated_t = torch.cat((tf1, tf2, tf3), dim=-1)
        t_input = concate(concatenated_t)
        diffusion_mean = diffusion_mu(t_input)
        model_output = model(x, diffusion_mean)  #训练模型输出的是噪声epsilon
        '''
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        '''
        def process_xstart(x):
            if denoised_fn is not None: #额外的函数处理
                x = denoised_fn(x)
            if clip_denoised:  #是否将结果限制到（-1，1）
                return x.clamp(-1, 1)
            return x
        #print(self.model_mean_type)
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start = pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        #x_t = x_t[:, :4, ...]
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
       
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
   
def generate_simplex_noise(
        Simplex_instance, x, t, random_param=False, octave=6, persistence=0.8, frequency=64,
        in_channels=32
        ):
    noise = torch.empty(x.shape).to(x.device)
    for i in range(in_channels):
        Simplex_instance.newSeed()
        if random_param:
            param = random.choice(
                    [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                     (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                     (2, 0.85, 8),
                     (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                     (1, 0.85, 8),
                     (1, 0.85, 4), (1, 0.85, 2), ]
                    )
            # 2D octaves seem to introduce directional artifacts in the top left
            noise[:, i, ...] = torch.unsqueeze(
                    torch.from_numpy(
                            # Simplex_instance.rand_2d_octaves(
                            #         x.shape[-2:], param[0], param[1],
                            #         param[2]
                            #         )
                            Simplex_instance.rand_3d_fixed_T_octaves(
                                    x.shape[-2:], t.detach().cpu().numpy(), param[0], param[1],
                                    param[2]
                                    )
                            ).to(x.device), 0
                    ).repeat(x.shape[0], 1, 1, 1)
        noise[:, i, ...] = torch.unsqueeze(
                torch.from_numpy(
                        # Simplex_instance.rand_2d_octaves(
                        #         x.shape[-2:], octave,
                        #         persistence, frequency
                        #         )
                        Simplex_instance.rand_3d_fixed_T_octaves(
                                x.shape[-2:], t.detach().cpu().numpy(), octave,
                                persistence, frequency
                                )
                        ).to(x.device), 0
                ).repeat(x.shape[0], 1, 1, 1)
    return noise

def reparameterize(mu, logvar): 
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std) 
        return mu + eps * std

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()
