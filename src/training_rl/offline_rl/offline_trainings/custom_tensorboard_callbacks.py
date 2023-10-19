from tensorboardX import SummaryWriter


class CustomSummaryWriter(SummaryWriter):
    """
    This is just a simple example of how to vreate your own custom wrapper
    """

    def __init__(self, log_dir, env):
        super().__init__(log_dir)
        self.env = env
        self.add_text("args", "ToDo: Add .... !!")

    def log_custom_info(self):
        info = 3
        # Log custom information from the environment
        self.add_scalar("custom_info/max_num_steps", info)
