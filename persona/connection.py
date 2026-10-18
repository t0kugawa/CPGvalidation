from dataclasses import dataclass, field
from typing import Dict, Any
from persona.parameter import PersonaParam


@dataclass
class Connection:
    """
    Synaptic connection specification connecting two populations.
    Supports optimizable parameters (weight, delay, etc.), STDP or static models,
    and optional noise recording.
    """
    source: str
    target: str
    model: str = "static_synapse"
    parameters: Dict[str, PersonaParam] = field(default_factory=dict)
    rule: str = "pairwise_bernoulli"
    p: float = 0.5
    noise_recorder_id: str | None = None
    extra_syn_spec: Dict[str, Any] = field(default_factory=dict)
    synapse_collection: Any = None
    key: str | None = None
    weight_recorder: Any = None

    def get_effective_p(self) -> float:
        return float(self.p)

    def sample_syn_spec(self, explored_params: Dict[str, float] | None = None) -> Dict[str, Any]:
        """
        Builds the syn_spec dictionary for nest.Connect.
        """
        spec: Dict[str, Any] = {"synapse_model": self.model}
        spec.update(self.extra_syn_spec)

        for param_name, param_obj in self.parameters.items():
            if explored_params is not None and param_name in explored_params:
                val = explored_params[param_name]
            else:
                val = param_obj.mean
            spec[param_name] = float(val)

        return spec
