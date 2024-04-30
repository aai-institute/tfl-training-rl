from enum import Enum

from training_rl.offline_rl.offline_policies.bcq_continuous_policy import (
    bcq_continuous_default_config, create_bcq_continuous_policy_from_dict)
from training_rl.offline_rl.offline_policies.bcq_discrete_policy import (
    bcq_discrete_default_config, create_bcq_discrete_policy_from_dict)
from training_rl.offline_rl.offline_policies.cql_continuous_policy import (
    cql_continuous_default_config, create_cql_continuous_policy_from_dict)
from training_rl.offline_rl.offline_policies.cql_discrete_policy import (
    cql_discrete_default_config, create_cql_discrete_policy_from_dict)
from training_rl.offline_rl.offline_policies.dagger_torcs_policy import dagger_torcs_default_config, \
    create_dagger_torcs_policy_from_dict
from training_rl.offline_rl.offline_policies.dqn_policy import (
    create_dqn_policy_from_dict, dqn_default_config)
from training_rl.offline_rl.offline_policies.il_policy import (
    create_il_policy_from_dict, il_default_config)
from training_rl.offline_rl.offline_policies.il_torcs_policy import il_torcs_default_config, \
    create_il_torcs_policy_from_dict
from training_rl.offline_rl.offline_policies.ppo_policy import (
    create_ppo_policy_from_dict, ppo_default_config)


class CallableEnum(Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class PolicyType(str, Enum):
    offline = "offline"
    onpolicy = "onpolicy"
    offpolicy = "offpolicy"


class PolicyName(str, Enum):
    bcq_discrete = "bcq_discrete"
    cql_continuous = "cql_continuous"
    imitation_learning = "imitation_learning"
    imitation_learning_torcs = "imitation_learning_torcs"
    dagger_torcs = "dagger_torcs"
    bcq_continuous = "bcq_continuous"
    cql_discrete = "cql_discrete"
    dqn = "dqn"
    ppo = "ppo"


class DefaultPolicyConfigFactoryRegistry(CallableEnum):
    bcq_discrete = bcq_discrete_default_config
    cql_continuous = cql_continuous_default_config
    imitation_learning = il_default_config
    bcq_continuous = bcq_continuous_default_config
    cql_discrete = cql_discrete_default_config
    dqn = dqn_default_config
    ppo = ppo_default_config
    imitation_learning_torcs = il_torcs_default_config
    dagger_torcs = dagger_torcs_default_config


class PolicyFactoryRegistry(CallableEnum):
    bcq_discrete = create_bcq_discrete_policy_from_dict
    cql_continuous = create_cql_continuous_policy_from_dict
    imitation_learning = create_il_policy_from_dict
    bcq_continuous = create_bcq_continuous_policy_from_dict
    cql_discrete = create_cql_discrete_policy_from_dict
    dqn = create_dqn_policy_from_dict
    ppo = create_ppo_policy_from_dict
    imitation_learning_torcs = create_il_torcs_policy_from_dict
    dagger_torcs = create_dagger_torcs_policy_from_dict



