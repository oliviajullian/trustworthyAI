"""
This file defines functions to generate data from a pre-defined 
Bayesian network together with conditional distributions for each node.
"""

from abc import ABC, abstractmethod

import networkx


class Sampler(ABC):
    @abstractmethod
    def __call__(self, parent_values: dict, *args, **kwargs):
        """
        The call method serves to sample from the conditional distribution of a node given that its parents take on the
        values defined in teh dictionary `parent_values`. The keys of `parent_values` should be the parent nodes,
        while the values associated to those keys should be the values that those nodes take.
        """
        pass


def generate_data_from_causal_network(causal_dag: networkx.DiGraph,
                                      conditional_distribution_samplers: dict[object, Sampler],
                                      no_samples: int = 1) -> list[dict]:
    """
    Given a Bayesian network, that is, a directed acyclic graph, together with a dictionary specifying the conditional
    distributions of a node given its parents, this function samples from the induced distribution.

    :param causal_dag: The directed acyclic graph underlying the Bayesian network, passed as a `networkx.Graph`.
    :param conditional_distribution_samplers: A dictionary whose keys are the vertices of `causal_dag`, and where each
     value associated to a key is an instance of the Sampler class. This instance of the sampler class should, when
     called, take as input the values of the parents of the node and sample from the conditional distribution of the
     node given its parents.
    :param no_samples: Number of samples to generate.
    :returns: A list of samples. Each sample is formatted as a dictionary whose keys are the nodes and whose values are
     the values of those nodes.
    """
    output = []
    nodes_ordered = list(networkx.topological_sort(causal_dag))

    for _ in range(no_samples):
        single_sample = {}
        for node in nodes_ordered:
            parents = list(causal_dag.predecessors(node))
            single_sample[node] = conditional_distribution_samplers[node](
                {parent: single_sample[parent] for parent in parents}
            )
        output.append(single_sample)

    return output


if __name__ == '__main__':
    import scipy, numpy

    # Example network: X -> Z <- Y
    G = networkx.DiGraph()
    G.add_node("X")
    G.add_node("Y")
    G.add_node("Z")

    G.add_edge("X", "Z")
    G.add_edge("Y", "Z")

    # Discrete example
    class SamplerXorY(Sampler):
        def __call__(self, parent_values: dict, *args, **kwargs):
            return scipy.stats.bernoulli(1/2).rvs()

    class SamplerZ(Sampler):
        def __call__(self, parent_values: dict, *args, **kwargs):
            propensity = parent_values["X"] + parent_values["Y"]
            return scipy.stats.bernoulli(1 / (1+numpy.exp(-propensity))).rvs()

    result = generate_data_from_causal_network(
        G,
        {"X": SamplerXorY(), "Y": SamplerXorY(), "Z": SamplerZ()},
        no_samples=1000
    )
