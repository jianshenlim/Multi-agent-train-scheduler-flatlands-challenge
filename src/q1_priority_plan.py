import copy
from datetime import datetime
import random

import numpy as np

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

'''
A* Implementation by
@author: Frits de Nijs
taken from assignment 1 solutions
'''
import numpy
import itertools
import heapq

from dataclasses import dataclass, field
from typing import Any

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv, RailEnvActions

# from testmaps import two_agent_collision_map, random_square_map
from numba.core.decorators import njit


class StateConverter:

    def __init__(self, env: RailEnv):
        self.width = env.rail.width
        self.num_tiles = env.rail.width * env.rail.height
        self.num_states = 4 * self.num_tiles

    def position_to_state(self, row, col, dir):
        return dir + 4 * col + 4 * self.width * row

    def position_to_tile(self, position):
        return position[1] + self.width * position[0]

    def state_to_position(self, state):
        dir = state % 4
        col = ((state - dir) / 4) % self.width
        row = ((state - dir - col * 4)) / (4 * self.width)
        return (row, col, dir)

    @staticmethod
    def state_to_tile(state):
        return numpy.int32((state - state % 4) / 4)


def convert_to_transition(env: RailEnv, conv: StateConverter):
    # Transition is a function: [state][action] -> new state
    transition = -numpy.ones((conv.num_states, 5), dtype=numpy.int32)

    # Action is valid in a particular state if it leads to a new position.
    valid_action = numpy.zeros((conv.num_states, 5), dtype=numpy.int32)

    for row in range(0, env.rail.height):
        for col in range(0, env.rail.width):
            for dir in range(0, 4):

                # Compute the current state index.
                state = conv.position_to_state(row, col, dir)

                # Compute the number of possible transitions.
                possible_transitions = env.rail.get_transitions(row, col, dir)
                num_transitions = numpy.count_nonzero(possible_transitions)

                if num_transitions > 0:

                    # The easy case: stop moving holds current state.
                    transition[state][RailEnvActions.STOP_MOVING] = state
                    valid_action[state][RailEnvActions.STOP_MOVING] = 1

                    # Forward is only possible in two cases, there is only 1 option, or
                    # the current direction can be maintained. Stop otherwise.
                    if num_transitions == 1:
                        new_direction = numpy.argmax(possible_transitions)
                        new_position = get_new_position((row, col), new_direction)
                        transition[state][RailEnvActions.MOVE_FORWARD] = conv.position_to_state(new_position[0],
                                                                                                new_position[1],
                                                                                                new_direction)
                        valid_action[state][RailEnvActions.MOVE_FORWARD] = 1
                    elif possible_transitions[dir] == 1:
                        new_position = get_new_position((row, col), dir)
                        transition[state][RailEnvActions.MOVE_FORWARD] = conv.position_to_state(new_position[0],
                                                                                                new_position[1], dir)
                        valid_action[state][RailEnvActions.MOVE_FORWARD] = 1
                    else:
                        transition[state][RailEnvActions.MOVE_FORWARD] = state

                    # Left is only possible if there is a transition out to the left of
                    # the current direction. Otherwise, we move like we would if going
                    # Forward.
                    new_direction = (dir - 1) % 4
                    if possible_transitions[new_direction]:
                        new_position = get_new_position((row, col), new_direction)
                        transition[state][RailEnvActions.MOVE_LEFT] = conv.position_to_state(new_position[0],
                                                                                             new_position[1],
                                                                                             new_direction)
                        valid_action[state][RailEnvActions.MOVE_LEFT] = transition[state][RailEnvActions.MOVE_LEFT] != \
                                                                        transition[state][RailEnvActions.MOVE_FORWARD]
                    else:
                        transition[state][RailEnvActions.MOVE_LEFT] = transition[state][RailEnvActions.MOVE_FORWARD]

                    # Right is only possible if there is a transition out to the Right of
                    # the current direction. Otherwise, we move like we would if going
                    # Forward.
                    new_direction = (dir + 1) % 4
                    if possible_transitions[new_direction]:
                        new_position = get_new_position((row, col), new_direction)
                        transition[state][RailEnvActions.MOVE_RIGHT] = conv.position_to_state(new_position[0],
                                                                                              new_position[1],
                                                                                              new_direction)
                        valid_action[state][RailEnvActions.MOVE_RIGHT] = transition[state][RailEnvActions.MOVE_RIGHT] != \
                                                                         transition[state][RailEnvActions.MOVE_FORWARD]
                    else:
                        transition[state][RailEnvActions.MOVE_RIGHT] = transition[state][RailEnvActions.MOVE_FORWARD]

    return (transition, valid_action)


@njit
def all_pairs_shortest_paths(num_states, transition):
    dist = numpy.ones((num_states, num_states), dtype=numpy.int32) * numpy.inf

    # Initialize; neighbors of the current state are at distance 1 step, current state at 0 steps.
    for state in range(0, num_states):
        for action in range(1, 4):
            next_state = transition[state][action]
            if next_state != -1 and next_state != state:
                dist[state][next_state] = 1
        dist[state][state] = 0

    # FLoyd-Warshall algorithm to compute distances of shortest paths.
    for k in range(0, num_states):
        for i in range(0, num_states):
            for j in range(0, num_states):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


class SearchEnv:
    def __init__(self, trainId, env: RailEnv):
        self.conv = StateConverter(env)
        model = convert_to_transition(env, self.conv)
        self.transition = model[0]
        self.valid_actions = model[1]
        self.shortest = all_pairs_shortest_paths(self.conv.num_states, self.transition)
        self.initial_state = numpy.zeros(len(env.agents), dtype=numpy.int32)
        self.initial_active = numpy.zeros(len(env.agents), dtype=numpy.int32)

        self.releasedate = env.agents[trainId].release_date         # added release date abd deadline along with max number of steps
        self.deadline = env.agents[trainId].deadline
        self.maxSteps = env._max_episode_steps

        self.agent = env.agents[trainId]
        self.initial_state[trainId] = self.conv.position_to_state(self.agent.initial_position[0], self.agent.initial_position[1],
                                                                self.agent.initial_direction)

        self.goal_tile = numpy.zeros(len(env.agents), dtype=numpy.int32)
        self.goal_tile[trainId] = self.conv.position_to_tile(env.agents[trainId].target)

        # Convert from tiles to states by adding directions 0 to 4.
        self.goal_states = numpy.mgrid[0:len(env.agents), 0:4][1] + self.goal_tile.reshape(len(env.agents), 1) * 4

    def get_root_node(self):
        initial_state = SearchState(self.initial_state.copy(), self.initial_active.copy())
        return SearchNode(0, None, None, self, initial_state,0)


class SearchState:
    def __init__(self, positions, actives):
        self.positions = positions
        self.actives = actives
        self.hash = hash(self.actives.tobytes()) + 31 * hash(self.positions.tobytes())

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return numpy.array_equal(self.actives, other.actives) and numpy.array_equal(self.positions, other.positions)
        else:
            return NotImplemented


@dataclass(order=True)
class SearchNode:
    f: int
    neg_g: int

    parent: Any = field(compare=False)
    action: Any = field(compare=False)
    searchenv: Any = field(compare=False)
    searchstate: Any = field(compare=False)

    def __init__(self, neg_g, parent, action, searchenv, searchstate,time):

        self.parent = parent
        self.action = action
        self.searchenv = searchenv
        self.searchstate = searchstate
        self.timeStep = time

        self.neg_g = neg_g - self.timing_penalty(searchenv.agent,time,searchenv.maxSteps)  # g(n) = g(n') + 1 + p(n)
        self.f = self.get_evaluation() + self.getFuturePenalty(searchenv.agent,time,searchenv.maxSteps)  # calculate future penalty occurred and add it to heuristic


    def __hash__(self):
        return self.searchstate.__hash__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.searchstate.__eq__(other.searchstate)
        else:
            return NotImplemented

    def agents_at_goal(self):
        """
        Check if the current state has all the agents at one of the goal states.
        """
        return self.searchenv.conv.state_to_tile(self.searchstate.positions) == self.searchenv.goal_tile

    def is_goal_state(self):
        return self.agents_at_goal().all()

    def get_evaluation(self):
        return -self.neg_g + self.get_heuristic()

    def get_heuristic(self):
        shortest_to_goal_states = self.searchenv.shortest[
            self.searchstate.positions.reshape(len(self.searchstate.positions), 1), self.searchenv.goal_states]
        shortest_to_goal_state = numpy.min(shortest_to_goal_states, 1)
        return numpy.int32(numpy.max(shortest_to_goal_state))

    def get_occupied_tiles(self):
        occupied = numpy.zeros(self.searchenv.conv.num_tiles)
        tiles = self.searchenv.conv.state_to_tile(self.searchstate.positions)
        valid_tiles = tiles[self.searchstate.actives == 1]
        occupied[valid_tiles] = 1
        return occupied

    def get_all_valid_actions(self):

        # Select, for each agent, the valid actions based on its position.
        agent_actions = self.searchenv.valid_actions[self.searchstate.positions]

        # Mask the rail transition actions for idle agents.
        agent_actions[self.searchstate.actives == 0] = [1, 0, 1, 0, 0]  # DO_NOTHING, or MOVE_FORWARD.

        # Mask the rail transition actions for done agents.
        agent_actions[self.agents_at_goal()] = [1, 0, 0, 0, 0]  # DO_NOTHING only.

        # Identify for each agent the IDs of the valid actions (i.e., [0, 1, 1, 0, 0] --> [1, 2])
        agent_action_list = [numpy.nonzero(a)[0] for a in agent_actions]

        # Return list containing for each agent, the IDs of the actions available to it.
        return itertools.product(*agent_action_list)

    def getFuturePenalty(self,agent, elapsedSteps ,max_episode_steps):
        """
        Calculate the total sum of potential future penalties
        """
        numberOfSteps = self.get_heuristic()  # get remaining distance need to travel
        totalPenalty = 0

        for x in range(numberOfSteps):  # for each remaining step calculate the penalty occurred from moving at that time
            penalty = self.timing_penalty(agent,elapsedSteps+x,max_episode_steps)
            totalPenalty += penalty

        return totalPenalty

    def timing_penalty(self,agent, elapsed_steps, max_episode_steps):

        penalty = 0

        if agent.status == RailAgentStatus.ACTIVE:
            steps_outside = 0

            if elapsed_steps <= agent.release_date:
                steps_outside = (1 + agent.release_date - elapsed_steps)
            if elapsed_steps >= agent.deadline:
                steps_outside = 1 + elapsed_steps - agent.deadline
            # Compute the normalized penalty.
            penalty = ((steps_outside * steps_outside) / (max_episode_steps * max_episode_steps / 4))
        return penalty

    def expand_node(self, actions ,trainno, occmap):

        """
        Input:
         - actions: an array, where actions[agent] is the action id that agent id will try to take.
        """
        # Determine which tiles are occupied now.
        occupied = self.get_occupied_tiles()
        occMapTimeMax = len(occmap) - 1

        # Make copy the current search state (to modify).
        new_states = self.searchstate.positions.copy()
        new_actives = self.searchstate.actives.copy()
        # Move agents in increasing order of their IDs.

        for i in range(0, len(self.searchstate.positions)):
            if (i != trainno):
                continue
            # Get the current state of the agent.
            current_state = new_states[i]
            current_tile = self.searchenv.conv.state_to_tile(current_state)
            current_position = self.searchenv.conv.state_to_position(current_state)

            blocked = False
            if occmap != [] and self.timeStep <= occMapTimeMax: # If train starts in a position occupied by another in the OCC map, expand fails
                if occmap[self.timeStep-1][int(current_position[0])][int(current_position[1])] == 1:
                    return None

            # Agent was inactive, wants to begin moving.
            if new_actives[i] == 0 and actions[i] == 2:
                if occmap != [] and self.timeStep <= occMapTimeMax:
                    if occmap[self.timeStep][int(current_position[0])][int(current_position[1])] == 1:  # if the train wants to become active but the current position is occupied, expand fails
                        blocked = True

                if occupied[current_tile] == 1 or blocked:
                    # Attempting to enter blocked tile, expand fails.
                    return None
                else:
                    # Activate agent, occupy tile.
                    new_actives[i] = 1
                    occupied[current_tile] = 1

            # Agent was active, attempt to apply action
            elif new_actives[i] == 1:

                # The agent is trying to move, so it frees up the current tile.
                occupied[current_tile] = 0

                next_state = self.searchenv.transition[current_state, actions[i]]
                next_tile = self.searchenv.conv.state_to_tile(next_state)

                next_position = self.searchenv.conv.state_to_position(next_state)   # get the next position of the train


                if occmap != [] and self.timeStep <= occMapTimeMax:      # If train''s next position is occupied by another in the OCC map, expand fails
                    if occmap[self.timeStep][int(next_position[0])][int(next_position[1])] == 1:
                        blocked = True

                if occmap != [] and self.timeStep + 1 <= occMapTimeMax:  # If the trains next position in the next time step is occupied, expand fails
                    if occmap[self.timeStep + 1][int(next_position[0])][int(next_position[1])] == 1:
                        return None

                if occupied[next_tile] == 1 or blocked:
                    # Attempting to enter blocked tile, expand fails.
                    return None
                else:
                    occupied[current_tile] = 0
                    occupied[next_tile] = 1
                    new_states[i] = next_state
                    # Goal state reached, remove the occupancy, deactivate.
                    if next_tile == self.searchenv.goal_tile[i]:
                        occupied[next_tile] = 0
                        new_actives[i] = 0
        return SearchNode(self.neg_g - 1, self, actions, self.searchenv, SearchState(new_states, new_actives), self.timeStep+1)

    def get_path(self):
        """
        Get path implementation is changed to now also returns a list of positions the train occupies when executing its actions
        """
        action_dict = dict(enumerate(self.action))
        position_dict = dict(enumerate(self.searchstate.positions))
        if self.parent.parent is None:
            return [action_dict],[position_dict]
        else:
            path,position = self.parent.get_path()
            path.append(action_dict)
            position.append(position_dict)
            return path,position


def a_star_search(root,i,occmap):
    # Count the number of expansions and generations.
    expansions = 0
    generations = 0

    # Open list is a priority queue over search nodes, closed set is a hash-based set for tracking seen nodes.
    openlist = []
    closed = set({root})

    # Initially, open list is just the root node.
    heapq.heappush(openlist, root)

    # While we have candidates to expand,
    while len(openlist) > 0:

        # Get the highest priority search node.
        current = heapq.heappop(openlist)
        # Increment number of expansions.
        expansions = expansions + 1

        # If we expand the goal node, we are done.
        if current.is_goal_state():
            path,positions = current.get_path()
            return (path, expansions, generations),positions

        # Otherwise, we will generate all child nodes.
        for action in current.get_all_valid_actions():
            # Create successor node from action.
            nextnode = current.expand_node(action,i,occmap)

            # Generated one more node.
            generations = generations + 1
            # If this is a valid new node, append it to the open list.
            # and not closed.__contains__(nextnode
            if nextnode is not None:
                closed.add(nextnode)
                heapq.heappush(openlist, nextnode)

    return (None, expansions, generations), None

def combine(schedule):
    """
    Combines a list of multiple action dicts for individual agents into a single dict for return to evaluation function
    """
    output=[]
    time = getLongestTime(schedule)
    for x in range (time):
        time={}
        for y in range(len(schedule)):
            agent =schedule[y]
            if len(agent)-1 >= x:
                time.update(agent[x])
            else:
                time.update({y:0})

        output.append(time)
    return output

def search_stats(env: RailEnv):
    """
    Search_stats generates a random schedule for prioritized planning, it calculates the occ map using a list of agent positions and builds a list of agent action dicts which it combines into a single dict for return to eval function
    """

    schedule = [0] * len(env.agents)
    mapPos = [0] * len(env.agents)

    random.seed(datetime.now())
    ordering = random.sample(range(0, len(env.agents)),len(env.agents)) # generates a random schedule for processing

    for i in ordering:  # for each agent in the ordering list
        occmap = computeMap(env.height,i,mapPos)    # generate occ map from the provided list of agent positions

        result,state = a_star_search(SearchEnv(i,env).get_root_node(),i,occmap) # perform A* search for single agent and calculate the result and a list of positions train occupies
        path = result[0]    # get the final action dict for result

        agentAction,agentPosition = cleanOutput(env,i,path,state)   # convert output into single agent format and agents state to a list of agents positions on execution its action

        schedule[i]= agentAction    # add schedule of agent to list of schedules
        mapPos[i] = agentPosition   # add the all of agent positions to the list of positions

    result = combine(schedule)  # combine all agents schedules into single dict for return to eval function

    return result


def search(env: RailEnv):
    return search_stats(env)

def cleanOutput(env,trainId,plan,state):
    """
    Clean the input values to those we can use, convert multi agent action dict to single agent action dict, convert list of states to list of y,x positions we can use
    """
    converter = StateConverter(env)
    for x in range(len(plan)):
        plan[x] = {trainId: plan[x][trainId]}
        position = converter.state_to_position(state[x][trainId])
        state[x] = {trainId: (int(position[0]), int(position[1]))}
    return plan,state

def getLongestTime(positions):
    """
    Obtain that agent with the longest schedule
    """
    length=0
    for agent in positions:
        if agent == 0:
            continue
        if (len(agent) > length):
            length = len(agent)
    return length


def computeMap(size,trainId, positions):
    """
    computes occupancy map of all trains before current trainId
    """
    occMap=[]
    timesteps = getLongestTime(positions)  # Get the longest train step time
    for i in range(timesteps):            # For each time step
        map = numpy.zeros((size, size), dtype=numpy.int32) # create occupancy map of env size
        for agent in range(len(positions)):             # For each agent
            agentPositions = positions[agent]             # Get current agent's list of positions
            if (agentPositions == 0 or len(agentPositions)-1<i): # if No positions (==0) or train is finished, skip
                continue
            else:
                if agent == trainId:    # if checking current train, ignore
                    continue
                elif agent < trainId: # if train is earlier than given train Id, takes up only one space on map
                    pos = agentPositions[i].get(agent)
                    y, x = pos[0],pos[1]
                    map[y][x] = 1
                else:
                    firstPos = agentPositions[i].get(agent) # else train takes up 2 spaces on map
                    y, x = firstPos[0], firstPos[1]
                    map[y][x] = 1
                    if not i-1<0:               # if not the very last move of schedule, take 2 spaces
                        SndPos = agentPositions[i-1].get(agent)
                        y, x = SndPos[0], SndPos[1]
                        map[y][x] = 1
        occMap.append(map)

    return occMap


