import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector:

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.EPSILON_START, args.EPSILON_FINISH, args.EPSILON_ANNEAL_TIME,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

        self.device = th.device(self.args.DEVICE)

    def update_epsilon(self, steps=0, train_steps=0):
        standard = 0
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            standard = steps
        elif self.args.EPSILON_UPDATE_STANDARD == "train_steps":
            standard = train_steps
        else:
            Exception("EPSILON_UPDATE_STANDARD is defined wrong")
        self.epsilon = self.schedule.eval(standard)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            self.update_epsilon(steps=t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        if self.args.ALGORITHM == 'coma':
            masked_policies[avail_actions == 0.0] = 0.0
            if test_mode and self.test_greedy:
                picked_actions = masked_policies.max(dim=2)[1]
            else:
                picked_actions = Categorical(masked_policies).sample().long()
        else:
            if t_env < self.args.START_TRAINING_EPISODE:
                random_actions = th.rand_like(masked_policies)
                picked_actions = random_actions.clamp(-self.args.MAX_ACTION, self.args.MAX_ACTION)
            else:
                if not test_mode:
                    masked_policies += th.normal(
                        0, self.args.MAX_ACTION * self.args.EXPL_NOISE,
                        size=masked_policies.shape
                    ).to(device=self.device)
                picked_actions = masked_policies.clamp(-self.args.MAX_ACTION, self.args.MAX_ACTION)
        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.EPSILON_START, args.EPSILON_FINISH, args.EPSILON_ANNEAL_TIME,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def update_epsilon(self, steps=0, train_steps=0):
        standard = 0
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            standard = steps
        elif self.args.EPSILON_UPDATE_STANDARD == "train_steps":
            standard = train_steps
        else:
            Exception("EPSILON_UPDATE_STANDARD is defined wrong")
        self.epsilon = self.schedule.eval(standard)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            self.update_epsilon(steps=t_env)

        if test_mode:
            # Greedy action selection only
            eps = 0.0
        else:
            eps = self.epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < eps).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
