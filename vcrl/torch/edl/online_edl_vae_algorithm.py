from vcrl.torch.skewfit.online_vae_algorithm import OnlineVaeAlgorithm

class OnlineEDLVaeAlgorithm(OnlineVaeAlgorithm):
    def __init__(self, *base_args, **base_kwargs):
        super().__init__(*base_args, **base_kwargs)

    def _train(self):
        # Add 'image_observation', 'image_achieved/desired_goal'
        # to enable computing exploration reward with image_achieved_goal.
        ob_keys_to_save = self.replay_buffer.ob_keys_to_save
        self.replay_buffer.ob_keys_to_save = [
            'image_observation',
            'image_desired_goal',
            'image_achieved_goal',
            'latent_observation',
            'latent_desired_goal',
            'latent_achieved_goal',
        ]
        super()._train()
        self.replay_buffer.ob_keys_to_save = ob_keys_to_save