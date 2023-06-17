"""
Implements the routing of qubits to a physical architecture, which is given as a parameter to
the routing function in form of an adjacency list.

Most of the functions are inspired by the paper "On the Qubit Routing Problem" by Cowtan et al.
"""
from .gate import Gate


def ph_qubit(mapping, i):
    """
    Returns the physical qubit to which the virtual qubit `i` is mapped.
    """
    return mapping[i]


def are_adjacent(mapping, adjacent_qubits, target, control):
    """
    Returns True if the virtal qubits `target` and `control` are adjacent in the current mapping.
    """
    return ph_qubit(mapping, target) in adjacent_qubits[ph_qubit(mapping, control)]


def swap_virt_qubits(gates, mapping, q0, q1):
    # add swap gate
    gates += Gate('swap', [ph_qubit(mapping, q0),
                  ph_qubit(mapping, q1)]).decompose()
    # update mapping
    mapping[q0], mapping[q1] = ph_qubit(mapping, q1), ph_qubit(mapping, q0)


def swap_ph_qubits(gates, mapping, q0, q1):
    # add swap gate
    gates += Gate('swap', [q0, q1]).decompose()
    # update mapping
    for i in range(len(mapping)):
        if mapping[i] == q0:
            mapping[i] = q1
        elif mapping[i] == q1:
            mapping[i] = q0


def find_virtual_neighbor(initial_routing_graph, mapping, virtual_qubit):
    for virtual_neighbors in initial_routing_graph[virtual_qubit]:
        if mapping[virtual_neighbors] == -1:
            return virtual_neighbors
    return None


def route(nr_ph_qubits, nr_virt_qubits, gates, adjacent_qubits):
    """
    Parameters:
        `nr_ph_qubits`: number of available physical qubits. Must be >= of the number of used virtual qubits.

        `nr_virt_qubits`: number of virtual qubits in the circuit.

        `gates`: a list of gates that need to be routed to the physical architecture. 
        The gates act between the virtual qubits.

        `adjacent_qubits`: maps each physical qubit to the list of physical qubits it is connected to. 
        This implementation only works with qubits that have bidirectional connections.

    Returns:
        The initial mapping of virtual qubits to physical qubits.

        The final mapping of virtual qubits to physical qubits.

        The routed circuit consisting of gates which act on the physical qubits and with some added swap operations.
    """
    if nr_ph_qubits < nr_virt_qubits:
        raise Exception(
            f"There are not enough physical qubits available. Virtual qubits: {nr_virt_qubits}, Physical qubits: {nr_ph_qubits}")
    # mapping[i] contains the physical qubit to which the virtual qubit i is mapped
    initial_mapping = find_initial_qubits_mapping(
        nr_ph_qubits, nr_virt_qubits, gates, adjacent_qubits)
    mapping, gates = add_necessary_swaps(
        nr_ph_qubits, gates, adjacent_qubits, initial_mapping)
    return initial_mapping, mapping, gates


def find_initial_qubits_mapping(nr_ph_qubits, nr_virt_qubits, gates, adjacent_qubits):
    """
    Inspired by https://drops.dagstuhl.de/opus/volltexte/2019/10397/pdf/LIPIcs-TQC-2019-5.pdf
    (Section 3.2 Initial Mapping)

    Parameters:
        `nr_ph_qubits`: number of available physical qubits. Must be >= of the number of used virtual qubits.

        `nr_virt_qubits`: number of virtual qubits in the circuit.

        `gates`: a list of gates that need to be routed to the physical architecture. 
        The gates act between the virtual qubits.

        `adjacent_qubits`: maps each physical qubit to the list of physical qubits it is connected to. 
        This implementation only works with qubits that have bidirectional connections.

    Returns:
        Some initial mapping of virtual qubits to physical qubits, found by trying to map the 
        qubit interaction graph to the physical architecture of the quantum computer.
        The qubit interaction graph only contains the first two interactions for each qubit, 
        such that the first interactions are priotitized in the routing (as described in the paper).
        The mapping is found by first mapping the highest connected physical qubit to the
        highest connected virtual qubit, and then repeatedly mapping the next neighbor of the two qubits to each other.
    """
    # this is the qubit interaction graph of the virtual qubits
    initial_routing_graph = [[] for _ in range(nr_virt_qubits)]
    # this is used to find the highest connected virtual qubit
    nr_neighbors_virt = [0 for _ in range(nr_virt_qubits)]

    for gate in gates:
        if gate.nr_qubits() == 2:
            q1 = gate.qubits[0]
            q2 = gate.qubits[1]
            nr_neighbors_virt[q1] += 1
            nr_neighbors_virt[q2] += 1
            # add edge to qubit interaction graph if it is one of the first two interactions for this qubit
            if len(initial_routing_graph[q1]) < 2 and len(initial_routing_graph[q2]) < 2:
                initial_routing_graph[q1].append(q2)
                initial_routing_graph[q2].append(q1)

    # map the initial_routing_graph to the physical architecture

    mapping = [-1 for _ in range(nr_virt_qubits)]
    already_mapped_physical = [False for _ in range(nr_ph_qubits)]
    # find virtual qubit with the most connections
    highest_degree_virt_qubit = nr_neighbors_virt.index(max(nr_neighbors_virt))
    # find physical qubit with the most connections
    nr_neighbors_ph = [len(neighbors) for neighbors in adjacent_qubits]
    highest_degree_ph_qubit = nr_neighbors_ph.index(max(nr_neighbors_ph))
    # start with highest degree qubits
    physical_qubit = highest_degree_ph_qubit
    # continue with this qubit in case there is no physical neighbor left to be mapped
    next_physical_qubit = 0
    virtual_qubit = highest_degree_virt_qubit
    # continue with this qubit in case there is no virtual neighbor left to be mapped
    next_virtual_qubit = 0
    # start again from highest degree qubits next time we reach the end of a chain
    already_tried_both_directions = False
    # map neighbors of current qubit to each other until all qubits are mapped
    while virtual_qubit < nr_virt_qubits:
        mapping[virtual_qubit] = physical_qubit
        already_mapped_physical[physical_qubit] = True
        # find a neighbor of the virtual qubit that wasn't mapped yet
        found_new_neighbor_virt = False
        virtual_qubit = find_virtual_neighbor(
            initial_routing_graph, mapping, virtual_qubit)
        found_new_neighbor_virt = (virtual_qubit != None)
        if not found_new_neighbor_virt and not already_tried_both_directions:
            # the first qubit to be mapped was highest_degree_virt_qubit, and we want to start by mapping both its neighbors
            virtual_qubit = find_virtual_neighbor(
                initial_routing_graph, mapping, highest_degree_virt_qubit)
            found_new_neighbor_virt = (virtual_qubit != None)
            if found_new_neighbor_virt:
                physical_qubit = highest_degree_ph_qubit
        # find a neighbor of the physical qubit that wasn't mapped yet and that has the highest degree
        found_new_neighbor_ph = False
        highest_degree = 0
        for physical_neighbors in adjacent_qubits[physical_qubit]:
            if not already_mapped_physical[physical_neighbors]:
                if len(adjacent_qubits[physical_neighbors]) > highest_degree:
                    physical_qubit = physical_neighbors
                    found_new_neighbor_ph = True
                    highest_degree = len(
                        adjacent_qubits[physical_neighbors])
        if not found_new_neighbor_ph:
            # find the next physical qubit that hasn't been mapped yet
            while True:
                if next_physical_qubit >= nr_ph_qubits:
                    # all physical qubits were already mapped
                    break
                if not already_mapped_physical[next_physical_qubit]:
                    physical_qubit = next_physical_qubit
                    next_physical_qubit += 1
                    break
                next_physical_qubit += 1
        if not found_new_neighbor_virt:
            # find the next virtual qubit that hasn't been mapped yet
            while True:
                if next_virtual_qubit >= nr_virt_qubits:
                    virtual_qubit = next_virtual_qubit  # so that the outer loop will also break
                    # all virtual qubits were already mapped
                    break
                if mapping[next_virtual_qubit] == -1:
                    virtual_qubit = next_virtual_qubit
                    next_virtual_qubit += 1
                    break
                next_virtual_qubit += 1
    # at this point all the virtual qubits are mapped to physical qubits
    return mapping


def add_necessary_swaps(nr_ph_qubits, gates, adjacent_qubits, initial_mapping):
    """
    Starting from the initial mapping, swap gates are added such that all CNOTs can be executed.
    The strategy is to find the shortest path between the two qubits that need to interact and
    then add swaps in order to move one of the qubits near the other one, following this shortest path.

    Returns:
        `mapping`: the new mapping of virtual qubits to physical qubits at the end of the circuit. 

        `output_gates`: the gates mapped to the physical architecture via the `mapping` and with 
        swap operations added in order to bring non-adjacent qubits to adjacent positions if a CNOT is performed between these two qubits.
    """
    mapping = initial_mapping.copy()
    output_gates = []
    # add all gates that can be directly executed directly to the output 
    # and add swaps for the gates that are not currently possible
    for gate in gates:
        if gate.nr_qubits() == 1 or (gate.nr_qubits() == 2 and are_adjacent(mapping, adjacent_qubits, gate.target(), gate.control()[0])):
            # can be safely added to the output
            output_gates.append(gate.map_qubits(mapping))
        elif gate.nr_qubits() > 2:
            raise Exception(
                f"Can't route gate acting between {gate.nr_qubits()} qubits. Maximal 2 qubits.")
        else:
            # add swaps to move the control qubit to the nearest physical qubit that is adjacent to the target qubit
            control = gate.control()[0]
            target = gate.target()
            path_to_qubit = find_path_to_nearest_connected_qubit(
                nr_ph_qubits, mapping, adjacent_qubits, target, control)
            # perfom the swaps to move control to the nearest connected qubit
            previous_qubit = path_to_qubit.pop(0)
            for current_qubit in path_to_qubit:
                swap_ph_qubits(output_gates, mapping,
                               current_qubit, previous_qubit)
                previous_qubit = current_qubit
            # now target and control are connected
            output_gates.append(gate.map_qubits(mapping))
    return mapping, output_gates


def find_path_to_nearest_connected_qubit(nr_ph_qubits, mapping, adjacent_qubits, target, control):
    """
    BFS starting from control qubit and ending when the target is found.
    Return the list of the shortest path between the control and the target.
    The target qubit is not added at the end of the list.
    Returns `None` if the target is not reachable from the control.

    The target and control qubits given as parameter are virtual qubits. 
    The function calculates the corresponding physical qubits and finds the path between the physical qubits.
    """
    # find the corresponding physical qubits
    target = ph_qubit(mapping, target)
    control = ph_qubit(mapping, control)
    # initialize BFS
    previous_qubit = [-1 for _ in range(nr_ph_qubits)]
    visited = [False for _ in range(nr_ph_qubits)]
    to_be_visited = [control]
    visited[control] = True
    # traverse architecture starting from control qubit
    while to_be_visited != []:
        current_qubit = to_be_visited.pop(0)
        found_target = False
        for neighbor in adjacent_qubits[current_qubit]:
            if not visited[neighbor]:
                to_be_visited.append(neighbor)
                previous_qubit[neighbor] = current_qubit
                visited[neighbor] = True
                if neighbor == target:
                    break
        if found_target:
            break
    if visited[target] == False:
        # target is not reachable from control
        return None

    # calculate path from control to target (without including the target)
    current_qubit = target
    path = []
    while current_qubit != control:
        current_qubit = previous_qubit[current_qubit]
        path.insert(0, current_qubit)
    return path


def slice_into_timesteps(gates):
    """
    Slice circuit into timesteps:
    The gates are divided in groups that can be executed simultaneously;
    single qubit gates are ignored.

    Inspired by
    https://drops.dagstuhl.de/opus/volltexte/2019/10397/pdf/LIPIcs-TQC-2019-5.pdf.

    Given that this step is not strictly necessary, I didn't use it in the final implementation of the routing algorithm.
    """

    sliced_circuit = []
    current_timestep = []
    used_qubits = []
    for gate in gates:
        if gate.nr_qubits() > 2:
            raise Exception(
                "Can't route gates involving more than two qubits. Please decompose the circuit first.")
        elif gate.nr_qubits() == 1:
            current_timestep.append(gate)
        elif gate.nr_qubits() == 2:
            currently_used_qubits = gate.qubits
            if gate.qubits[0] in used_qubits or gate.qubits[1] in used_qubits:
                # add the gate to the next timestep
                sliced_circuit.append(current_timestep)
                current_timestep = []
                used_qubits = currently_used_qubits
            else:
                # add the gate to the current timestep
                current_timestep.append(gate)
                used_qubits += gate.qubits
    return sliced_circuit
