"""
This module contains the core classes for the hierarqcal package. Qmotif is the base class for all motifs, Qmotifs is a sequence of motifs and Qhierarchy is the full quantum circuit architecture, that handles the interaction between motifs.

Create a hierarchy as follows:

.. code-block:: python

    from hierarqcal import Qinit, Qconv, Qpool
    my_qcnn = Qinit(8) + (Qconv(1) + Qpool(pattern="right")) * 3

The above creates a hierarchy that resembles a reverse binary tree architecture. There are 8 free qubits and then a convolution-pooling unit is repeated three times.
"""

from collections.abc import Sequence
from enum import Enum
import warnings
from copy import copy, deepcopy
from collections import deque, namedtuple
import numpy as np
import itertools as it
import sympy as sp


class Primitive_Types(Enum):
    """
    Enum for primitive types
    """

    CYCLE = "cycle"
    MASK = "mask"
    PERMUTE = "permute"
    FREE = "free"


class Qunitary:
    """
    Base class for all unitary operations
    """

    def __init__(self, function=None, n_symbols=0, arity=2, symbols=None):
        """
        Args:
            function (function, optional): Function to apply, if None then the default function is used. Defaults to None.
            n_symbols (int, optional): Number of symbols that function use, if None then the default number of symbols is used. Defaults to None.
            arity (int, optional): Number of qubits that function acts upon, two means 2-qubit unitaries, three means 3-qubits unitaries and so on. Defaults to 2.
            symbols (tuple(Parameter), optional): Tuple of symbol values (rotation angles). Defaults to None.
        """
        self.function = function
        self.n_symbols = n_symbols
        self.arity = arity
        self.symbols = None
        self.edge = None

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def get_symbols(self):
        return self.symbols

    def set_symbols(self, symbols=None):
        if len(symbols) != self.n_symbols:
            raise ValueError(
                f"Number of symbols must be {self.n_symbols} for this function"
            )
        self.symbols = symbols

    def set_edge(self, edge):
        self.edge = edge


class Default_Mappings(Enum):
    """
    Enum for default mappings
    """

    CYCLE = Qunitary(n_symbols=1, arity=2)
    MASK = None
    PERMUTE = Qunitary(n_symbols=1, arity=2)


class Qmotif:
    """
    Quantum Circuit architectures are created by stacking motifs hierarchically, the lowest level motifs (primitives) are building blocks for higher level ones.
    Examples of primitives are convolution (:py:class:`Qconv`), pooling (:py:class:`Qpool`) and dense (:py:class:`Qdense`) operations which inherits this class. Each motif is a directed graph with nodes
    Q for qubits and edges E for unitary operations applied between them, the direction of an edge being the order of interaction for the unitary. Each instance has
    pointers to its predecessor and successor.This class is for a single motif and the Qmotifs class is for a sequence of motifs stored as tuples, then sequences of
    motifs are again stored as tuples  This is to allow hierarchical stacking which in the end is one tuple of motifs.
    """

    def __init__(
        self,
        Q=[],
        E=[],
        Q_avail=[],
        edge_order=[1],
        next=None,
        prev=None,
        mapping=None,
        symbols=None,
        is_default_mapping=True,
        is_operation=True,
        share_weights=True,
    ) -> None:
        """
        # TODO add description of args especially the new mapping arg

        Args:
            arity (int, optional): Number of qubits per unitary (nodes per edge), two means 2-qubit unitaries, three means 3-qubits unitaries and so on. Defaults to 2.
            mapping tuple(tuple(function, int) or Qhierarchy): Either a tuple containing the function to execute with it's number of parameters or a Qhierarchy consisting of just one operational motif
                  tuple: the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.
        """
        # Meta information
        self.is_operation = is_operation
        self.is_default_mapping = is_default_mapping
        # graph
        self.Q = Q
        self.Q_avail = Q_avail
        self.E = E
        self.edge_order = edge_order
        # higher level graph
        self.mapping = mapping
        self.symbols = symbols
        self.share_weights = share_weights
        self.edge_mapping = None  # TODO docstring, edgemapping only gets set when the motif gets edges generated
        self.n_symbols = 0
        # pointers
        self.prev = prev
        self.next = next
        # Handle mappings
        if self.is_default_mapping:
            self.arity = 2
        else:
            if isinstance(self.mapping, Qhierarchy):
                # If mapping was specified as sub-hierarchy, convert it to a qunitary
                new_mapping = Qunitary(
                    function=None,
                    n_symbols=self.mapping.n_symbols,
                    arity=len(self.mapping.tail.Q),
                )
                new_mapping.function = self.mapping.get_unitary_function()
                self.mapping = new_mapping
            self.arity = self.mapping.arity

    def __add__(self, other):
        """
        Add two motifs together, this  appends them next to each other in a tuple.

        Args:
            other (Qmotif): Motif to append to current motif (self).

        Returns:
            Qmotifs(tuple): A 2-tuple of motifs: (self, other) in the order they were added. The tuples contain copies of the original motifs.
        """
        return self.append(other)

    def __mul__(self, other):
        """
        Repeat motif "other" times.

        Args:
            other (int): Number of times to repeat motif.

        Returns:
            Qmotifs(tuple): A tuple of motifs: (self, self, ..., self), where each is a new object copied from the original.
        """
        # TODO must create new each time investigate __new__, this copies the object.
        return Qmotifs((deepcopy(self) for i in range(other)))

    def append(self, other):
        """
        Append motif to current motif.

        Args:
            other (Qmotif): Motif to append to current motif (self).

        Returns:
            Qmotifs(tuple): A 2-tuple of motifs: (self, other) in the order they were added. The tuples contain copies of the original motifs.
        """
        return Qmotifs((deepcopy(self), deepcopy(other)))

    def set_Q(self, Q):
        """
        Set the qubit labels Q of the motif.

        Args:
            Q (list(int or string)): List of qubit labels.
        """
        self.Q = Q

    def set_E(self, E):
        """
        Set the edges E of the motif.

        Args:
            E (list(tuples)): List of edges.
        """
        if 0 in self.edge_order:
            # Raise error if 0 is in edge order
            raise ValueError(
                "Edge order can't contain 0, as there is no 0th edge. Use 1 instead, edge order is based on ordering, that is [1] means first edge comes first [2,8] means second edge comes first, then 8th edge comes second. There is no 0th edge"
            )
        E_ordered = [E[i - 1] for i in self.edge_order if i - 1 < len(E)]
        E_rest = [edge for edge in E if edge not in E_ordered]
        Ep_l = E_ordered + E_rest
        self.E = E

    def set_arity(self, arity):
        """
        Set the arity of the motif (qubits per unitary/ nodes per edge).

        Args:
            arity (int): Number of qubits per unitary
        """
        self.arity = arity

    def set_edge_order(self, edge_order):
        """
        Set the edge order of the motif (order of unitaries applied).

        Args:
            edge_order (list(int)): List of edge orders.
        """
        self.edge_order = edge_order

    def set_Qavail(self, Q_avail):
        """
        Set the available qubits Q_avail of the motif. This is usually calculated by the previous motif in the stack, for example pooling removes qubits from being available and
        would use this function to update the available qubits after it's action.

        Args:
            Q_avail (list): List of available qubits.
        """
        self.Q_avail = Q_avail

    def set_mapping(self, mapping):
        """
        Specify the unitary operations applied according to the type of motif.

        Args:
            TODO update to Qunitary
            mapping (tuple(function, int)): Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.
        """
        self.mapping = mapping

    def set_next(self, next):
        """
        Set the next motif in the stack.

        Args:
            next (Qmotif): Next motif in the stack.
        """
        self.next = next

    def set_prev(self, prev):
        """
        Set the previous motif in the stack.

        Args:
            prev (Qmotif): Previous motif in the stack.
        """
        self.prev = prev

    def set_share_weights(self, share_weights):
        """
        Set the share_weights flag.

        Args:
            share_weights (bool): Whether to share weights within a motif.
        """
        self.share_weights = share_weights

    def set_is_operation(self, is_operation):
        """
        Set the is_operation flag.

        Args:
            is_operation (bool): Whether the motif is an operation.
        """
        self.is_operation = is_operation

    def get_symbols(self):
        if not (self.edge_mapping is None):
            if self.share_weights:
                yield from self.edge_mapping[0].get_symbols()
            else:
                for edge_map in self.edge_mapping:
                    yield from edge_map.get_symbols()

    def set_symbols(self, symbols=None, start_idx=0):
        """
        Set the symbol's.

        Args:
            symbols (tuple(sympy.Symbols))
        """
        if not (self.mapping is None):
            if symbols is None:
                symbols = sp.symbols(
                    f"x_{start_idx}:{start_idx + self.mapping.n_symbols*(len(self.E) if not(self.share_weights) else 1)}"
                )
            else:
                if len(symbols) != self.mapping.n_symbols * (
                    len(self.E) if not (self.share_weights) else 1
                ):
                    raise ValueError(
                        f"Number of symbols {len(symbols)} does not match number of symbols in motif {self.mapping.n_symbols*(len(self.E) if not(self.share_weights) else 1)}"
                    )
            self.edge_mapping = []
            idx = 0
            for edge in self.E:
                tmp_mapping = deepcopy(self.mapping)
                tmp_mapping.set_edge(edge)
                tmp_mapping.set_symbols(symbols[idx : idx + self.mapping.n_symbols])
                if not (self.share_weights):
                    idx += self.mapping.n_symbols
                self.edge_mapping.append(tmp_mapping)
            self.n_symbols = len(symbols)


class Qmotifs(tuple):
    """
    A tuple of motifs, this is the data structure for storing sequences motifs. It subclasses tuple, so all tuple methods are available.
    """

    # TODO mention assumption that only operators should be used i.e. +, *
    # TODO explain this hackery, it's to ensure the case (a,b)+b -> (a,b,c) no matter type of b
    def __add__(self, other):
        """
        Add two tuples of motifs together.

        Args:
            other (Qmotifs or Qmotif): Multiple motifs or singe motif to add to current sequence of motifs.

        Returns:
            Qmotifs(tuple): A single tuple of the motifs that were added, These are copies of the original motifs, since tuples are immutable.
        """
        if isinstance(other, Sequence):
            return Qmotifs(tuple(self) + tuple(other))
        else:
            return Qmotifs(tuple(self) + (other,))

    def __mul__(self, other):
        """
        Repeat motifs "other" times.

        Args:
            other (int): Number of times to repeat motifs.

        Returns:
            Qmotifs(tuple): A tuple of motifs: (self, self, ..., self), where each is a new object copied from the original.

        Raises:
            ValueError: Only integers are allowed for multiplication.
        """
        # repeats "other=int" times i.e. other=5 -> i in range(5)
        if type(other) is int:
            return Qmotifs((deepcopy(item) for i in range(other) for item in self))
        else:
            raise ValueError("Only integers are allowed for multiplication")


class Qcycle(Qmotif):
    """
    A convolution motif, used to specify convolution operations in the quantum neural network.
    TODO implement open boundary for dense and pooling also + tests
    """

    def __init__(
        self,
        stride=1,
        step=1,
        offset=0,
        boundary="periodic",
        **kwargs,
    ):
        """
        TODO determine if boundary is the best name for this, or if it should be called something else.
        TODO docstring for edge order, importantly it's based on ordering, that is [1] means first edge comes first [2,8] means second edge comes first, then 8th edge comes second. There is no 0th edge
        Initialize a convolution motif.

        Args:
            stride (int, optional): Stride of the convolution. Defaults to 1.
            step (int, optional): Step of the convolution. Defaults to 1.
            offset (int, optional): Offset of the convolution. Defaults to 0.

        """
        self.type = Primitive_Types.CYCLE
        self.stride = stride
        self.step = step
        self.offset = offset
        self.boundary = boundary
        # Specify sequence of gates:
        mapping = kwargs.get("mapping", None)
        is_default_mapping = True if mapping is None else False
        # motif_symbols = None # TODO maybe allow symbols to be intialised
        # Initialize graph
        super().__init__(is_default_mapping=is_default_mapping, **kwargs)

    def __call__(self, Qc_l, *args, **kwargs):
        """
        Call the motif, this generates the edges and qubits of the motif (directed graph) based on it's available qubits.
        Each time a motif in the stack changes, a loop runs through the stack from the beginning and calls each motif to update the graph (the available qubits, the edges etc).

        Args:
            Qc_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments, such as:

                * mapping (tuple(function, int)): TODO update this docstring, it's not a tuple anymore.
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
            Qconv: Returns the updated version of itself, with correct nodes and edges.
        """
        # Determine cycle operation
        nq_available = len(Qc_l)
        if self.stride % nq_available == 0:
            # TODO make this clear in documentation
            # warnings.warn(
            #     f"Stride and number of available qubits can't be the same, received:\nstride: {self.stride}\n available qubits:{nq_available}. Defaulting to stride of 1"
            # )
            self.stride = 1
        if self.boundary == "open":
            mod_nq = lambda x: x % nq_available
            Ec_l = [
                tuple(
                    (
                        Qc_l[i + j * self.stride]
                        for j in range(self.arity)
                        if i + j * self.stride < nq_available
                    )
                )
                for i in range(self.offset, nq_available, self.step)
            ]
            # Remove all that is not "complete"
            Ec_l = [edge for edge in Ec_l if len(edge) == self.arity]

        else:
            mod_nq = lambda x: x % nq_available
            Ec_l = [
                tuple((Qc_l[mod_nq(i + j * self.stride)] for j in range(self.arity)))
                for i in range(self.offset, nq_available, self.step)
            ]
            # Remove all that is not "complete", i.e. contain duplicates
            Ec_l = [edge for edge in Ec_l if len(set(edge)) == self.arity]
        if (
            len(Ec_l) == self.arity
            and sum([len(set(Ec_l[0]) - set(Ec_l[k])) == 0 for k in range(self.arity)])
            == self.arity
        ):
            # If there are only as many edges as qubits, and they are the same, then we can keep only one of them
            Ec_l = [Ec_l[0]]
        self.set_Q(Qc_l)
        # Set order of edges
        # if 0 in self.edge_order: TODO check if this is removable
        #     # Raise error if 0 is in edge order
        #     raise ValueError(
        #         "Edge order can't contain 0, as there is no 0th edge. Use 1 instead, edge order is based on ordering, that is [1] means first edge comes first [2,8] means second edge comes first, then 8th edge comes second. There is no 0th edge"
        #     )
        # Ec_l_ordered = [Ec_l[i - 1] for i in self.edge_order if i - 1 < len(Ec_l)]
        # Ec_l_rest = [edge for edge in Ec_l if edge not in Ec_l_ordered]
        # Ec_l = Ec_l_ordered + Ec_l_rest
        # All qubits are still available for the next operation
        self.set_Qavail(Qc_l)
        mapping = kwargs.get("mapping", None)
        if mapping:
            self.set_mapping(mapping)
        # It is important that set_E gets called last, as sets of symbol creation for the motif
        self.set_E(Ec_l)
        start_idx = kwargs.get("start_idx", 0)
        self.set_symbols(start_idx=start_idx)

        return self

    def __eq__(self, other):
        if isinstance(other, Qcycle):
            self_attrs = vars(self)
            other_attrs = vars(other)
            for attr, value in self_attrs.items():
                if attr not in other_attrs or other_attrs[attr] != value:
                    return False

            return True
        return False


class Qpermute(Qmotif):
    """
    A dense motif, it connects unitaries to all possible combinations of qubits (all possible edges given Q) in the quantum circuit.
    """

    def __init__(self, combinations=True, **kwargs):
        self.type = Primitive_Types.PERMUTE
        self.combinations = combinations
        # Specify sequence of gates:
        mapping = kwargs.get("mapping", None)
        is_default_mapping = True if mapping is None else False
        # motif_symbols = None
        # if mapping is None:
        #     # default convolution layer is defined as U with 1 parameter.
        #     is_default_mapping = True
        #     # Default mapping is a unitary with one parameter, TODO generalize, if default changes we might want to change this
        #     motif_symbols = sp.symbols(f"x_{0}:{1}")
        # else:
        #     is_default_mapping = False
        #     if isinstance(mapping, Qhierarchy):
        #         motif_symbols = mapping.symbols
        #     else:
        #         motif_symbols = sp.symbols(f"x_{0}:{mapping[1]}")
        # kwargs["symbols"] = motif_symbols
        # Initialize graph
        super().__init__(is_default_mapping=is_default_mapping, **kwargs)

    def __call__(self, Qc_l, *args, **kwargs):
        """
        Call the motif, this is used to generate the edges and qubits of the motif (directed graph) based on it's available qubits.
        Each time a motif in the stack changes, a loop runs through the stack from the beginning and calls each motif to update the graph (the available qubits, the edges etc).

        Args:
            Qc_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments, such as:

                * mapping (tuple(function, int)):
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
            Qdense: Returns the updated version of itself, with correct nodes and edges.
        """
        # All possible wire combinations
        if self.combinations:
            Ec_l = list(it.combinations(Qc_l, r=self.arity))
        else:
            Ec_l = list(it.permutations(Qc_l, r=self.arity))
        if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
            Ec_l = [Ec_l[0]]
        self.set_Q(Qc_l)
        # Set order of edges
        # if 0 in self.edge_order: TODO check if this is removable
        #     # Raise error if 0 is in edge order
        #     raise ValueError(
        #         "Edge order can't contain 0, as there is no 0th edge. Use 1 instead, edge order is based on ordering, that is [1] means first edge comes first [2,8] means second edge comes first, then 8th edge comes second. There is no 0th edge"
        #     )
        # Ec_l_ordered = [Ec_l[i - 1] for i in self.edge_order if i - 1 < len(Ec_l)]
        # Ec_l_rest = [edge for edge in Ec_l if edge not in Ec_l_ordered]
        # Ec_l = Ec_l_ordered + Ec_l_rest
        # All qubits are still available for the next operation
        self.set_Qavail(Qc_l)
        mapping = kwargs.get("mapping", None)
        if mapping:
            self.set_mapping(mapping)
        # It is important that set_E gets called last, as sets of symbol creation for the motif
        self.set_E(Ec_l)
        start_idx = kwargs.get("start_idx", 0)
        self.set_symbols(start_idx=start_idx)
        return self

    def __eq__(self, other):
        if isinstance(other, Qpermute):
            self_attrs = vars(self)
            other_attrs = vars(other)
            for attr, value in self_attrs.items():
                if attr not in other_attrs or other_attrs[attr] != value:
                    return False
            return True
        return False


class Qmask_Base(Qmotif):
    def __init__(
        self,
        pattern="right",  # nearest_circle, nearest_tower, nearest_square, right, left, up, down, lambda function
        **kwargs,
    ):
        """
        TODO Provide topology for nearest neighbor pooling., options, circle, tower, square
        TODO Open, Periodic boundary
        """
        self.type = Primitive_Types.MASK
        self.pattern = pattern
        super().__init__(**kwargs)

    def __call__(self, Q, E, remaining_q, is_operation, **kwargs):
        mapping = kwargs.get("mapping", None)
        start_idx = kwargs.get("start_idx", 0)
        self.set_Q(Q)
        self.set_Qavail(remaining_q)
        if mapping:
            self.set_mapping(mapping)
        self.set_is_operation(is_operation)
        self.set_E(E)
        self.set_symbols(start_idx=start_idx)

    def get_mask_pattern_fn(self, mask_pattern, Qp_l=[]):
        """
        Get the pattern function for the pooling operation.

        Args:
            mask_pattern (str or lambda): The pattern type, can be "left", "right", "even", "odd", "inside" or "outside" which corresponds to a specific pattern (see comments in code below).
                                            The string can also be a bit string, i.e. "01000" which pools the 2nd qubit.
                                            If a lambda function is passed, it is used as the pattern function, it should work as follow: mask_pattern_fn([0,1,2,3,4,5,6,7]) -> [0,1,2,3], i.e.
                                            the function returns a sublist of the input list based on some pattern. What's nice about passing a function is that it can be list length independent,
                                            meaning the same kind of pattern will be applied as the list grows or shrinks.
            Qp_l (list): List of available qubits.
        """
        if isinstance(mask_pattern, str):
            # Mapping words to the pattern type
            if mask_pattern == "left":
                # 0 1 2 3 4 5 6 7
                # x x x x
                mask_pattern_fn = lambda arr: arr[0 : len(arr) // 2 : 1]
            elif mask_pattern == "right":
                # 0 1 2 3 4 5 6 7
                #         x x x x
                mask_pattern_fn = lambda arr: arr[len(arr) // 2  : len(arr): 1]
            elif mask_pattern == "even":
                # 0 1 2 3 4 5 6 7
                # x   x   x   x
                mask_pattern_fn = lambda arr: arr[0::2]
            elif mask_pattern == "odd":
                # 0 1 2 3 4 5 6 7
                #   x   x   x   x
                mask_pattern_fn = lambda arr: arr[1::2]
            elif mask_pattern == "inside":
                # 0 1 2 3 4 5 6 7
                #     x x x x
                mask_pattern_fn = (
                    lambda arr: arr[
                        len(arr) // 2
                        - len(arr) // 4 : len(arr) // 2
                        + len(arr) // 4 : 1
                    ]
                    if len(arr) > 2
                    else [arr[1]]
                )  # inside
            elif mask_pattern == "outside":
                # 0 1 2 3 4 5 6 7
                # x x         x x
                mask_pattern_fn = (
                    lambda arr: [
                        item
                        for item in arr
                        if not (
                            item
                            in arr[
                                len(arr) // 2
                                - len(arr) // 4 : len(arr) // 2
                                + len(arr) // 4 : 1
                            ]
                        )
                    ]
                    if len(arr) > 2
                    else [arr[0]]
                )  # outside
            else:
                # Assume pattern is in form contains a string specifying which indices to remove
                # For example "01001" removes idx 1 and 4 or qubit 2 and 5
                # The important thing here is for pool pattern to be the same length as the current number of qubits
                if len(mask_pattern) == len(Qp_l):
                    mask_pattern_fn = lambda arr: [
                        item
                        for item, indicator in zip(arr, mask_pattern)
                        if indicator == "1"
                    ]
                else:
                    # Attempt to use the pattern as a base pattern
                    # TODO explain in docs and maybe print a warning
                    # For example "101" will be used as "10110110" if there are 8 qubits
                    if any("*" == c for c in mask_pattern):
                        # Wildcard pattern
                        n_ones = mask_pattern.count("1")
                        n_zeros = len(Qp_l) - n_ones
                        if n_zeros <= 0:
                            # We go as far as we can with a pattern and then fill everything with 0s if we can't fit the pattern
                            base = "0" * len(Qp_l)
                        else:
                            base = mask_pattern.replace("*", "0" * n_zeros)
                        mask_pattern_fn = lambda arr: [
                            item
                            for item, indicator in zip(arr, base)
                            if indicator == "1"
                        ]
                    else:
                        # If there are no wildcard characters, then we assume that the pattern is a base pattern
                        # and we will repeat it until it is the same length as the current number of qubits
                        base = mask_pattern * (len(Qp_l) // len(mask_pattern))
                        base = base[: len(Qp_l)]
                        mask_pattern_fn = lambda arr: [
                            item
                            for item, indicator in zip(arr, base)
                            if indicator == "1"
                        ]

        else:
            mask_pattern_fn = mask_pattern
        return mask_pattern_fn


class Qmask(Qmask_Base):
    """
    A masking motif, it masks qubits based on some pattern TODO some controlled operation where the control is not used for the rest of the circuit).
    This motif changes the available qubits for the next motif in the stack.
    """

    def __init__(
        self,
        pattern="right",  # nearest_circle, nearest_tower, nearest_square, right, left, up, down, lambda function
        stride=0,
        step=1,
        offset=0,
        boundary="periodic",
        **kwargs,
    ):
        """
        TODO Provide topology for nearest neighbor pooling., options, circle, tower, square
        TODO Open, Periodic boundary
        """
        self.stride = stride
        self.step = step
        self.offset = offset
        self.boundary = boundary
        mapping = kwargs.get("mapping", None)
        is_default_mapping = True if mapping is None else False
        # Initialize graph
        super().__init__(pattern, is_default_mapping=is_default_mapping, **kwargs)

    def __call__(self, Qp_l, *args, **kwargs):
        """
        Call the motif, this is used to generate the edges and qubits of the motif (directed graph) based on it's available qubits.
        Each time a motif in the stack changes, a loop runs through the stack from the beginning and calls each motif to update the graph (the available qubits, the edges etc).

        Args:
            Qp_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments, such as:

                * mapping (tuple(function, int)):
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
            Qpool: Returns the updated version of itself, with correct nodes and edges.
        """
        # The idea is to mask qubits based on some pattern
        # This can be done with or without applying a unitary. Applying a unitary "preserves" their information (usually through some controlled unitary)
        # This enables patterns of: pooling in quantum neural networks, coarse graining or entanglers or just plain masking
        # The logic is as follows:
        # First check if there are more than 1 qubit available, since we don't want to make our last available qubit unavailable.
        # Then check if there is a associated untitary, if there is:
        #   If arity is 2 we have some predifined patterns + the general pattern functionality
        #       Predifined: right, left, inside, outside, even, odd, nearest_circle, nearest_tower
        #   If arity is more, then we just have general pattern functionality
        # General pattern works as follows:
        # Provided a binary string of length arity, concatenate it to itself until it is of length len(Qp_l)
        # Then use this binary string to mask the qubits, where 1 means mask and 0 means keep
        # Stride, Step, Offset manages the connectivity of masked and unmasked qubits, generally we want unmasked ones to be the target
        # of masked ones, so that we enable deffered measurement.
        # The most general usage is providing your own filter function
        # TODO add this to docs and explain how to provide own filter function.
        is_operation = False
        if len(Qp_l) > 1:
            # Check if predifined pattern was provided
            self.mask_pattern_fn = self.get_mask_pattern_fn(self.pattern, Qp_l)
            measured_q = self.mask_pattern_fn(Qp_l)
            remaining_q = [q for q in Qp_l if not (q in measured_q)]
            Ep_l = []
            # Check if there is a unitary associated with the motif
            if not (self.mapping is None):
                is_operation = True
                # All below generates edges for associated unitaries
                if isinstance(self.pattern, str) and not (
                    all((c in ("0", "1") for c in self.pattern))
                ):
                    if len(remaining_q) > 0:
                        # TODO add nearest neighbor modulo nq
                        if self.pattern == "nearest_circle":
                            Ep_l = [
                                (
                                    Qp_l[Qp_l.index(i)],
                                    min(
                                        remaining_q,
                                        key=lambda x: abs(Qp_l.index(i) - Qp_l.index(x))
                                        % len(remaining_q)
                                        // 2,
                                    ),
                                )
                                for i in measured_q
                            ]
                        elif self.pattern == "nearest_tower":
                            Ep_l = [
                                (
                                    Qp_l[Qp_l.index(i)],
                                    min(
                                        remaining_q,
                                        key=lambda x: abs(
                                            Qp_l.index(i) - Qp_l.index(x)
                                        ),
                                    ),
                                )
                                for i in measured_q
                            ]
                        else:
                            Ep_l = [
                                (
                                    measured_q[i],
                                    remaining_q[(i + self.stride) % len(remaining_q)],
                                )
                                for i in range(len(measured_q))
                            ]
                    else:
                        # No qubits were pooled
                        Ep_l = []
                        remaining_q = Qp_l
                else:
                    # General pattern functionality:
                    # TODO maybe generalize better arity > 2, currently my idea is that the pattern string should completely
                    # specify the form of the n qubit unitary, that is length of pattern string should equal arity.
                    if isinstance(self.pattern, str):
                        if len(self.pattern) != self.arity:
                            raise ValueError(
                                f"Pattern string should be of length arity {self.arity}, if it is a string."
                            )
                        nq_available = len(Qp_l)
                    if self.stride % nq_available == 0:
                        self.stride = 1  # TODO test if this is neccesary
                    # We generate edges the same way as convolutions
                    if self.boundary == "open":
                        mod_nq = lambda x: x % nq_available
                        Ep_l = [
                            tuple(
                                (
                                    Qp_l[i + j * self.stride]
                                    for j in range(self.arity)
                                    if i + j * self.stride < nq_available
                                )
                            )
                            for i in range(self.offset, nq_available, self.step)
                        ]
                        # Remove all that is not "complete"
                        Ep_l = [edge for edge in Ep_l if len(edge) == self.arity]

                    else:
                        mod_nq = lambda x: x % nq_available
                        Ep_l = [
                            tuple(
                                (
                                    Qp_l[mod_nq(i + j * self.stride)]
                                    for j in range(self.arity)
                                )
                            )
                            for i in range(self.offset, nq_available, self.step)
                        ]
                        # Remove all that is not "complete", i.e. contain duplicates
                        Ep_l = [edge for edge in Ep_l if len(set(edge)) == self.arity]
                    if (
                        len(Ep_l) == self.arity
                        and sum(
                            [
                                len(set(Ep_l[0]) - set(Ep_l[k])) == 0
                                for k in range(self.arity)
                            ]
                        )
                        == self.arity
                    ):
                        # If there are only as many edges as qubits, and they are the same, then we can keep only one of them
                        Ep_l = [Ep_l[0]]
                    # Then we apply the pattern to record which edges go away
                    self.mask_pattern_fn = self.get_mask_pattern_fn(self.pattern, Qp_l)
                    measured_q = [
                        qubit for edge in Ep_l for qubit in self.mask_pattern_fn(edge)
                    ]
                    remaining_q = [q for q in Qp_l if not (q in measured_q)]

        else:
            # raise ValueError(
            #     "Pooling operation not added, Cannot perform pooling on 1 qubit"
            # )
            # No qubits were pooled
            # TODO make clear in documentation, no pooling is done if 1 qubit remain
            Ep_l = []
            remaining_q = Qp_l
        super().__call__(
            Q=Qp_l, E=Ep_l, remaining_q=remaining_q, is_operation=is_operation, **kwargs
        )
        return self

    def __eq__(self, other):
        if isinstance(other, Qmask):
            self_attrs = vars(self)
            other_attrs = vars(other)

            for attr, value in self_attrs.items():
                if not (attr == "mask_pattern_fn"):
                    if attr not in other_attrs or other_attrs[attr] != value:
                        return False

            return True
        return False


class Qunmask(Qmask_Base):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        TODO possibility to give masking motif to undo
        """
        # Initialize graph
        super().__init__(*args, **kwargs)

    def __call__(self, Qp_l, *args, **kwargs):
        """
        TODO
        """
        q_initial = kwargs.get("q_initial", None)
        is_operation = False
        Ep_l = []
        self.mask_pattern_fn = self.get_mask_pattern_fn(self.pattern, q_initial)
        unmasked_q = self.mask_pattern_fn(q_initial)
        unique_unmasked = [q for q in unmasked_q if q not in Qp_l]
        new_avail_q = Qp_l + unique_unmasked
        super().__call__(
            Q=Qp_l, E=Ep_l, remaining_q=new_avail_q, is_operation=is_operation, **kwargs
        )
        return self


class Qhierarchy:
    """
    The main class that manages the "stack" of motifs, it handles the interaction between successive motifs and when a motifs are added to the stack,
    it updates all the others accordingly. An object of this class fully captures the architectural information of a hierarchical quantum circuit.
    It also handles function (unitary operation) mappings.
    """

    def __init__(self, qubits, function_mappings={}) -> None:
        # Set available qubit
        if isinstance(qubits, Qmotif):
            self.tail = qubits
            self.head = self.tail
        else:
            self.tail = Qinit(qubits)
            self.head = self.tail
        self.function_mappings = function_mappings
        self.mapping_counter = {
            primitive_type.value: 1 for primitive_type in Primitive_Types
        }
        self.n_symbols = 0

    def __add__(self, other):
        """
        Add a motif, motifs or hierarchy to the stack.

        Args:
            other (Qmotif, Qhierarchy, Sequence(Qmotif)): The motif, motifs or hierarchy to add to the stack.

        Returns:
            Qhierarchy: A new Qhierarchy object with the motif(s) added to the stack.
        """
        if isinstance(other, Qhierarchy):
            new_qcnn = self.merge(other)
        elif isinstance(other, Sequence):
            if isinstance(other[-1], Qmotif):
                new_qcnn = self.extend(other)
            elif isinstance(other[-1], Qhierarchy):
                new_qcnn = self.extmerge(other)
        elif isinstance(other, Qmotif):
            new_qcnn = self.append(other)
        return new_qcnn

    def __mul__(self, other):
        """
        Repeat the hierarchy a number of times. If a motif(s) is provided, it is added to the stack.

        Args:
            other (int, Qmotif, Sequence(Qmotif)): The number of times to repeat the hierarchy or the motif(s) to add to the stack.

        Returns:
            Qhierarchy: A new Qhierarchy object with the motif(s) added to the stack.
        """
        # TODO
        if isinstance(other, Qhierarchy):
            new_qcnn = self.merge(other)
        elif isinstance(other, Sequence):
            new_qcnn = self.extend(other)
        elif isinstance(other, Qmotif):
            new_qcnn = self.append(other)
        elif isinstance(other, int):
            t = (self,) * other
            if other > 1:
                new_qcnn = t[0] + t[1:]
            else:
                # TODO no action for multiplication too large, might want to make use of 0
                new_qcnn = self
        return new_qcnn

    def __iter__(self):
        """
        Generator to go from head to tail and only return operations (motifs that correspond to operations).
        """
        # Generator to go from head to tail and only return operations
        current = self.tail
        while current is not None:
            if current.is_operation:
                yield current
            current = current.next

    def __repr__(self) -> str:
        description = ""
        current = self.tail
        while current is not None:
            if current.is_operation:
                description += f"{repr(vars(current))}\n"
            current = current.next
        return description

    def __eq__(self, other):
        if isinstance(other, Qhierarchy):
            for layer_self, layer_other in zip(self, other):
                if not (layer_self == layer_other):
                    return False
            return True
        return False

    def __getitem__(self, index):
        current = self.tail
        for i in range(index + 1):
            if i == index:
                return current
            current = current.next
        return None

    # TODO remove
    # def update_symbols(self, motif, n_symbols=None, n_parent_unitaries=None):
    #     mapping = motif.mapping
    #     current_symbol_count = len(self.symbols)
    #     if n_symbols is None:
    #         # This is the normal case
    #         if mapping is None:
    #             n_symbols = getattr(default_symbol_counts, motif.type)
    #         else:
    #             n_symbols = mapping[1]
    #         if motif.share_weights == True:
    #             new_symbols = sp.symbols(
    #                 f"x_{current_symbol_count}:{current_symbol_count+n_symbols}"
    #             )
    #         else:
    #             n_unitaries = len(motif.E)
    #             new_symbols = sp.symbols(
    #                 f"x_{current_symbol_count}:{current_symbol_count+n_unitaries*n_symbols}"
    #             )
    #         motif.set_symbols(new_symbols)
    #         self.symbols = self.symbols + new_symbols
    #     else:
    #         # This is the case where a sub Qhierarchy was used as a mapping
    #         if motif.share_weights == True:
    #             new_symbols = sp.symbols(
    #                 f"x_{current_symbol_count}:{current_symbol_count+n_symbols}"
    #             )
    #         else:
    #             # n_map_symbols is the number of symbols that the subQhierarchy motif used
    #             new_symbols = sp.symbols(
    #                 f"x_{current_symbol_count}:{current_symbol_count+n_parent_unitaries*n_symbols}"
    #             )
    #         motif.set_symbols(new_symbols)
    #         self.symbols = self.symbols + new_symbols

    # def init_symbols(self, values=None, uniform_range=(0, 2 * np.pi)):
    #     # If values is None, then the values will be initialise uniformly based on the range given in init_uniform
    #     # If values is a dictionary, then the keys should be the symbols and the values should be the values to substitute
    #     # If values is a list/array/tuple, then the values will be substituted in the order of self.symbols

    #     if values is None:
    #         values = np.random.uniform(*uniform_range, size=len(self.symbols))
    #         self.symbols = values
    #     elif isinstance(values, dict):
    #         values = self.symbols.subs(values)
    #     else:
    #         # First check that the length of values is the same as the number of symbols
    #         if len(values) != len(self.symbols):
    #             raise ValueError(
    #                 f"values must be the same length as the number of symbols ({len(self.symbols)})"
    #             )
    #         # Substitute the values in the order of self.symbols
    #         self.symbols = values
    #     # Update the symbols in each motif
    #     cur_symbol_count = 0
    #     for primitive in self:
    #         primitive.symbols = self.symbols[
    #             cur_symbol_count : cur_symbol_count + len(primitive.symbols)
    #         ]
    #         # motif.set_symbols(self.symbols[cur_symbol_count:cur_symbol_count + len(motif.symbols)])
    #         cur_symbol_count = cur_symbol_count + len(primitive.symbols)

    # def update_motif_symbols(self, motif, values=None, uniform_range=(0, 2 * np.pi)):
    #     # Symbols need to be numeric at this point, TODO add test, or find a better way to handle them

    #     if values is None:
    #         values = np.random.uniform(*uniform_range, size=len(motif.symbols))
    #         motif.set_symbols(values)
    #     elif isinstance(values, (list, tuple, np.ndarray)):
    #         # First check that the length of values is the same as the number of symbols
    #         if len(values) != len(motif.symbols):
    #             raise ValueError(
    #                 f"values must be the same length as the number of symbols ({len(motif.symbols)})"
    #             )
    #         # Substitute the values in the order of self.symbols
    #         motif.set_symbols(values)
    #     else:
    #         # TODO tensor needs to be valid also
    #         motif.set_symbols(values)

    #     # Update overall symbols based on motifs
    #     cur_symbol_count = 0
    #     for tmp_motif in self:
    #         self.symbols[
    #             cur_symbol_count : cur_symbol_count + len(tmp_motif.symbols)
    #         ] = tmp_motif.symbols
    #         cur_symbol_count = cur_symbol_count + len(tmp_motif.symbols)

    def get_symbols(self):
        return (symbol for layer in self for symbol in layer.get_symbols())

    def set_symbols(self, symbols):
        idx = 0
        for layer in self:
            n_symbols = len([_ for _ in layer.get_symbols()])
            layer.set_symbols(symbols[idx : idx + n_symbols])
            idx += n_symbols

    def get_unitary_function(self):
        """
        Convert the Qhierarchy into a function that can be called.
        """

        def unitary_function(bits, symbols=None):
            self.update_Q(bits)
            if not (symbols is None):
                self.set_symbols(symbols)
            for layer in self:
                for unitary in layer.edge_mapping:
                    unitary.function(unitary.edge, unitary.symbols)

        return unitary_function

    def append(self, motif):
        """
        Add a motif to the stack of motifs and update it (call to generate nodes and edges) according to the action of the previous motifs in the stack.

        Args:
            motif (Qmotif): The motif to add to the stack.

        Returns:
            Qhierarchy: A new Qhierarchy object with the new motif added to the stack.
        """
        motif = deepcopy(motif)

        if motif.is_operation & motif.is_default_mapping:
            mapping = None
            mappings = self.function_mappings.get(motif.type, None)
            if mappings:
                # If function mapping was provided
                mapping = mappings[
                    (self.mapping_counter.get(motif.type) - 1) % len(mappings)
                ]
                self.mapping_counter.update(
                    {motif.type: self.mapping_counter.get(motif.type) + 1}
                )
                motif.is_default_mapping = False
            else:
                # If no function mapping was provided
                mapping = getattr(Default_Mappings, motif.type.name).value
            motif(self.head.Q_avail, mapping=mapping, q_initial=self.tail.Q)
            # self.update_symbols(motif) TODO
        # elif motif.is_operation & isinstance(motif.mapping, Qhierarchy):
        #     """
        #     The goal is to map a parents edges like motif.E = [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7), (6, 7, 8), (7, 8, 9), (8, 9, 1), (9, 1, 2)]
        #     to the edges of the motif we want to build it from, like sub_Qhierarchy.head.E = [(1, 2), (1, 3), (2, 3)]
        #     First we need to generate the motif with a arity equal to the tail of the sub_Qhierarchy, then we need to map the edges of the parent motif to the edges of the child motif
        #     and finally we need to update the parent motif with the new edges and nodes, it's arity ends up with the same as the arity of the child motif.
        #     What should the parent inherit from the child? The function mapping, arity and weights and sub types.
        #     """
        #     # Mapping is a Qhierarchy object, we only need to check the first element of the tuple, as it's either a tuple of tuples (for function mapping) or a tuple of Qhierarchys (for Qhierarchy mapping)
        #     # TODO assumption only subQhierarchy only has 1 layer, 1 motif i.e. a Qinit tail and a Qmotif Head

        #     # Copy sub Qhierarchy
        #     sub_hierarchy = deepcopy(
        #         motif.mapping
        #     )  # TODO need to do this in a loop for multiple sub hierarchies
        #     # TODO test sub_hierarchy has less qubits than the parent hierarchy
        #     motif.arity = len(sub_hierarchy.tail.Q_avail)
        #     # Generate edges with arity based on sub_hierarchy tail (the total number of qubits the sub hierarchy acts on)
        #     motif(self.head.Q_avail, mapping=sub_hierarchy.head.mapping)
        #     n_parent_unitaries = len(motif.E)
        #     E_parent = np.array(motif.E)
        #     E_child = np.array(sub_hierarchy.head.E)
        #     # Get the corresponding indices since the nodes are just labels
        #     E_child_idx = np.searchsorted(sub_hierarchy.head.Q_avail, E_child)
        #     E_parent_new = [
        #         tuple(parent_edge[idx])
        #         for parent_edge in E_parent
        #         for idx in E_child_idx
        #     ]
        #     # Change motifs arity to sub_hierarchys layer
        #     motif.arity = sub_hierarchy.head.arity
        #     # Update symbols
        #     motif(self.head.Q_avail)
        #     # overwrite parent edges with new edges
        #     motif.is_default_mapping = sub_hierarchy.head.is_default_mapping
        #     motif.set_mapping(sub_hierarchy.head.mapping)
        #     motif.set_E(deepcopy(E_parent_new))
        #     motif.sub_type = sub_hierarchy.head.type
        #     # self.update_symbols(
        #     #     motif,
        #     #     n_symbols=len(sub_hierarchy.symbols),
        #     #     n_parent_unitaries=n_parent_unitaries,
        #     # ) TODO
        else:
            n_symbols = len([_ for _ in self.get_symbols()])
            motif(self.head.Q_avail, start_idx=n_symbols, q_initial=self.tail.Q)
            # self.update_symbols(motif) TODO

        new_hierarchy = deepcopy(self)
        new_hierarchy.head.set_next(motif)
        new_hierarchy.head = new_hierarchy.head.next
        new_hierarchy.n_symbols = len([_ for _ in new_hierarchy.get_symbols()])
        return new_hierarchy

    def extend(self, motifs):
        """
        Add and update a list of motifs to the current stack of motifs call each to generate their nodes and edges according to the action of the previous motifs in the stack.

        Args:
            motifs (Qmotifs): A tuple of motifs to add to the stack.

        Returns:
            Qhierarchy: A new Qhierarchy object with the motifs added to the stack.
        """
        new_hierarchy = deepcopy(self)
        for motif in motifs:
            new_hierarchy = new_hierarchy.append(motif)
        return new_hierarchy

    def merge(self, hierarchy):
        """
        Merge two Qhierarchy objects by adding the head of the second one to the tail of the first one. Both are copied to ensure immutability.

        Args:
            hierarchy (Qhierarchy): The Qhierarchy object to merge with the current one.

        Returns:
            Qhierarchy: A new Qhierarchy object with the two merged.
        """
        # ensure immutability
        other_hierarchy = deepcopy(hierarchy)
        new_hierarchy = deepcopy(self)
        other_hierarchy.update_Q(
            new_hierarchy.head.Q_avail, start_idx=new_hierarchy.n_symbols
        )
        new_hierarchy.head.set_next(other_hierarchy.tail)
        new_hierarchy.head = other_hierarchy.head
        return new_hierarchy

    def extmerge(self, hierarchies):
        """
        Merge a list of Qhierarchy objects by adding the head of each one to the tail of the previous one. All are copied to ensure immutability.

        Args:
            hierarchies (Qhierarchy): A list of Qhierarchy objects to merge with the current one.

        Returns:
            Qhierarchy: A new Qhierarchy object with the list merged.
        """
        new_qcnn = deepcopy(self)
        for hierarchy in hierarchies:
            new_qcnn = new_qcnn.merge(hierarchy)
        return new_qcnn

    def update_Q(self, Q, start_idx=0):
        """
        Update the number of available qubits for the hierarchy and update the rest of the stack accordingly.

        Args:
            Q (list(int or string)): The list of available qubits.
        """
        motif = self.tail(Q)
        while motif.next is not None:
            motif = motif.next(motif.Q_avail, start_idx=start_idx, q_initial=self.tail.Q)
            start_idx += motif.n_symbols

    def copy(self):
        """
        Returns:
            Qhierarchy: A copy of the current Qhierarchy object.
        """
        return deepcopy(self)


class Qinit(Qmotif):
    """
    Qinit motif, represents a freeing up qubit for the QCNN, that is making qubits available for future operations. All Qhierarchy objects start with a Qinit motif.
    It is a special motif that has no edges and is not an operation.
    """

    def __init__(self, Q, **kwargs) -> None:
        if isinstance(Q, Sequence):
            Qinit = Q
        elif type(Q) == int:
            Qinit = [i + 1 for i in range(Q)]
        self.type = Primitive_Types.FREE
        # Initialize graph
        super().__init__(Q=Qinit, Q_avail=Qinit, is_operation=False, **kwargs)

    def __add__(self, other):
        """
        Add a motif, motifs or hierarchy to the stack with self.Qinit available qubits.
        """
        return Qhierarchy(self) + other

    def __call__(self, Q, *args, **kwargs):
        """
        Calling TODO add explanation just returns the object. Kwargs and Args are ignored, it just ensures that Qinit can be called the same way operational motifs can.
        """
        self.set_Q(Q)
        self.set_Qavail(Q)
        return self


# %%
