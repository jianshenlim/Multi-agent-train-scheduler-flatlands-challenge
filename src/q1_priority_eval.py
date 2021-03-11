import time
import threading

import numpy
import pyglet

from flatland.utils.rendertools import RenderTool
from flatland.utils.graphics_pgl import RailViewWindow

from utils.environment_utils import create_multi_agent_environment
from q1_priority_plan import search
# from backUp import search

# Evaluates the A* search algorithm over a number of samples.
def evalfun(num_samples = 100, timed=True, debug = False, refresh = 0.1):

    # A list of (mapsize, agent count) tuples, change or extend this to test different sizes.
    problemsizes = [(5, 3), (7, 4), (9, 5), (11, 6), (13, 7),(15,8),(17,9),(19,10)]
    # problemsizes = [(25, 4)]
    # problemsizes = [(13,10)]
    # Create a list of seeds to consider.
    seeds = numpy.random.randint(2**29, size=3*num_samples)
    # seeds = [265145549]
    # seeds = [305190040, 103788521, 268612265, 51654794, 287029697, 295171606, 431758321,
     # 353059089 , 89285543 , 46456488, 214148915, 530337246, 432131843,   9183764,
     # 216441307]
    # print(seeds)

    print("%10s\t%8s\t%8s\t%9s" % ("Dimensions", "Success", "Rewards", "Runtime"))
    for problemsize in problemsizes:
        j = 0
        for _ in range(0, num_samples):

            # Create environments while they are not the intended dimension.
            env = create_multi_agent_environment(problemsize[0], problemsize[1], timed, seeds[j])
            j = j + 1
            while len(env.agents) != problemsize[1]:
                env = create_multi_agent_environment(problemsize[0], problemsize[1], timed, seeds[j])
                j = j + 1

            # Create a renderer only if in debug mode.
            if debug:
                env_renderer = RenderTool(env, screen_width=1920, screen_height=1080)

            # Time the search.
            start = time.time()
            schedule = search(env)
            duration = time.time() - start;
    
            if debug:
                env_renderer.render_env(show=True, frames=False, show_observations=False)
                time.sleep(refresh)
        
            # Validate that environment state is unchanged.
            assert env.num_resets == 1 and env._elapsed_steps == 0
        
            # Run the schedule
            success = False
            sumreward = 0
            for action in schedule:
                # print("ENV STEP ",env._elapsed_steps)
                _, _reward_dict, _done, _ = env.step(action)
                success = all(_done.values())
                # print(_reward_dict.values())
                sumreward = sumreward + sum(_reward_dict.values())
                if debug:
                    #print(action)
                    env_renderer.render_env(show=True, frames=False, show_observations=False)
                    time.sleep(refresh)
        
            # Print the performance of the algorithm
            print("%10s\t%8s\t%8.3f\t%9.6f\t%i" % (str(problemsize), str(success), sumreward, duration,len(schedule)))


if __name__ == "__main__":

    # Number of maps of each size to consider.
    _num_maps = 1
    # If _timed = true, impose release dates and deadlines. False for regular (Assignment 1) behavior.
    _timed = True

    _debug = True
    _refresh = 0.3

    if (_debug):
        window = RailViewWindow()

    evalthread = threading.Thread(target=evalfun, args=(_num_maps,_timed,_debug,_refresh,))
    evalthread.start()

    if (_debug):
        pyglet.clock.schedule_interval(window.update_texture, 1/120.0)
        pyglet.app.run()

    evalthread.join()
