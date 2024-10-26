"""
Back ported methods: call, set_attr from v0.26
Disabled auto-reset after done
Added render method.
"""


import numpy as np
import multiprocessing
import time
import sys
from enum import Enum
from copy import deepcopy
import traceback
from typing import Any, Callable, Sequence

from multiprocessing import Queue
from multiprocessing.connection import Connection

from gymnasium import Space, logger
from gymnasium.vector.vector_env import VectorEnv
from gymnasium.core import ActType, Env, ObsType, RenderFrame
from gymnasium.vector.async_vector_env import AsyncVectorEnv, _async_worker, AsyncState
from gymnasium.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
)

from gymnasium.core import ObsType
from gymnasium.vector.utils import (
    create_shared_memory,
    create_empty_array,
    batch_space,
    write_to_shared_memory,
    read_from_shared_memory,
    concatenate,
    CloudpickleWrapper,
    clear_mpi_env_vars,
)
from torch import mul

__all__ = ["AsyncVectorEnv"]



class CustomAsyncVectorEnv(AsyncVectorEnv):
    """Vectorized environment that runs multiple environments in parallel.

    It uses ``multiprocessing`` processes, and pipes for communication.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("Pendulum-v1", num_envs=2, vectorization_mode="async")
        >>> envs
        AsyncVectorEnv(Pendulum-v1, num_envs=2)
        >>> envs = gym.vector.AsyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> envs
        AsyncVectorEnv(num_envs=2)
        >>> observations, infos = envs.reset(seed=42)
        >>> observations
        array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32)
        >>> infos
        {}
        >>> _ = envs.action_space.seed(123)
        >>> observations, rewards, terminations, truncations, infos = envs.step(envs.action_space.sample())
        >>> observations
        array([[-0.1851753 ,  0.98270553,  0.714599  ],
               [ 0.6193494 ,  0.7851154 , -1.0808398 ]], dtype=float32)
        >>> rewards
        array([-2.96495728, -1.00214607])
        >>> terminations
        array([False, False])
        >>> truncations
        array([False, False])
        >>> infos
        {}
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        dummy_env_fn = None,
        shared_memory: bool = True,
        copy: bool = True,
        context: str | None = None,
        daemon: bool = True,
        worker: (
            Callable[
                [int, Callable[[], Env], Connection, Connection, bool, Queue], None
            ]
            | None
        ) = None,
        observation_mode: str | Space = "same",
    ):
        """Vectorized environment that runs multiple environments in parallel.

        Args:
            env_fns: Functions that create the environments.
            shared_memory: If ``True``, then the observations from the worker processes are communicated back through
                shared variables. This can improve the efficiency if the observations are large (e.g. images).
            copy: If ``True``, then the :meth:`AsyncVectorEnv.reset` and :meth:`AsyncVectorEnv.step` methods
                return a copy of the observations.
            context: Context for `multiprocessing`. If ``None``, then the default context is used.
            daemon: If ``True``, then subprocesses have ``daemon`` flag turned on; that is, they will quit if
                the head process quits. However, ``daemon=True`` prevents subprocesses to spawn children,
                so for some environments you may want to have it set to ``False``.
            worker: If set, then use that worker in a subprocess instead of a default one.
                Can be useful to override some inner vector env logic, for instance, how resets on termination or truncation are handled.
            observation_mode: Defines how environment observation spaces should be batched. 'same' defines that there should be ``n`` copies of identical spaces.
                'different' defines that there can be multiple observation spaces with different parameters though requires the same shape and dtype,
                warning, may raise unexpected errors. Passing a ``Tuple[Space, Space]`` object allows defining a custom ``single_observation_space`` and
                ``observation_space``, warning, may raise unexpected errors.

        Warnings:
            worker is an advanced mode option. It provides a high degree of flexibility and a high chance
            to shoot yourself in the foot; thus, if you are writing your own worker, it is recommended to start
            from the code for ``_worker`` (or ``_async_worker``) method, and add changes.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
            ValueError: If observation_space is a custom space (i.e. not a default space in Gym,
                such as gymnasium.spaces.Box, gymnasium.spaces.Discrete, or gymnasium.spaces.Dict) and shared_memory is True.
        """
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        self.observation_mode = observation_mode

        self.num_envs = len(env_fns)

        # This would be nice to get rid of, but without it there's a deadlock between shared memory and pipes
        # Create a dummy environment to gather the metadata and observation / action space of the environment
        if dummy_env_fn is None:
            dummy_env = env_fns[0]()
        else:
            dummy_env = dummy_env_fn()

        # As we support `make_vec(spec)` then we can't include a `spec = dummy_env.spec` as this doesn't guarantee we can actual recreate the vector env.
        self.metadata = dummy_env.metadata
        self.render_mode = dummy_env.render_mode

        self.single_action_space = dummy_env.action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        if isinstance(observation_mode, tuple) and len(observation_mode) == 2:
            assert isinstance(observation_mode[0], Space)
            assert isinstance(observation_mode[1], Space)
            self.observation_space, self.single_observation_space = observation_mode
        else:
            if observation_mode == "same":
                self.single_observation_space = dummy_env.observation_space
                self.observation_space = batch_space(
                    self.single_observation_space, self.num_envs
                )
            elif observation_mode == "different":
                # the environment is created and instantly destroy, might cause issues for some environment
                # but I don't believe there is anything else we can do, for users with issues, pre-compute the spaces and use the custom option.
                env_spaces = [env().observation_space for env in self.env_fns]

                self.single_observation_space = env_spaces[0]
                self.observation_space = batch_differing_spaces(env_spaces)
            else:
                raise ValueError(
                    f"Invalid `observation_mode`, expected: 'same' or 'different' or tuple of single and batch observation space, actual got {observation_mode}"
                )

        dummy_env.close()
        del dummy_env

        # Generate the multiprocessing context for the observation buffer
        ctx = multiprocessing.get_context(context)
        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    self.single_observation_space, _obs_buffer, n=self.num_envs
                )
            except CustomSpaceError as e:
                raise ValueError(
                    "Using `AsyncVector(..., shared_memory=True)` caused an error, you can disable this feature with `shared_memory=False` however this is slower."
                ) from e
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = worker or _async_worker
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_spaces()

    def call_each(self, name: str, 
            args_list: list=None, 
            kwargs_list: list=None, 
            timeout = None):
        n_envs = len(self.parent_pipes)
        if args_list is None:
            args_list = [[]] * n_envs
        assert len(args_list) == n_envs

        if kwargs_list is None:
            kwargs_list = [dict()] * n_envs
        assert len(kwargs_list) == n_envs

        # send
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for i, pipe in enumerate(self.parent_pipes):
            pipe.send(("_call", (name, args_list[i], kwargs_list[i])))
        self._state = AsyncState.WAITING_CALL

        # receive
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results
    

# class AsyncVectorEnv(VectorEnv):
#     """Vectorized environment that runs multiple environments in parallel. It
#     uses `multiprocessing` processes, and pipes for communication.
#     Parameters
#     ----------
#     env_fns : iterable of callable
#         Functions that create the environments.
#     observation_space : `gym.spaces.Space` instance, optional
#         Observation space of a single environment. If `None`, then the
#         observation space of the first environment is taken.
#     action_space : `gym.spaces.Space` instance, optional
#         Action space of a single environment. If `None`, then the action space
#         of the first environment is taken.
#     shared_memory : bool (default: `True`)
#         If `True`, then the observations from the worker processes are
#         communicated back through shared variables. This can improve the
#         efficiency if the observations are large (e.g. images).
#     copy : bool (default: `True`)
#         If `True`, then the `reset` and `step` methods return a copy of the
#         observations.
#     context : str, optional
#         Context for multiprocessing. If `None`, then the default context is used.
#         Only available in Python 3.
#     daemon : bool (default: `True`)
#         If `True`, then subprocesses have `daemon` flag turned on; that is, they
#         will quit if the head process quits. However, `daemon=True` prevents
#         subprocesses to spawn children, so for some environments you may want
#         to have it set to `False`
#     worker : function, optional
#         WARNING - advanced mode option! If set, then use that worker in a subprocess
#         instead of a default one. Can be useful to override some inner vector env
#         logic, for instance, how resets on done are handled. Provides high
#         degree of flexibility and a high chance to shoot yourself in the foot; thus,
#         if you are writing your own worker, it is recommended to start from the code
#         for `_worker` (or `_worker_shared_memory`) method below, and add changes
#     """
#
#     def __init__(
#         self,
#         env_fns,
#         dummy_env_fn=None,
#         observation_space=None,
#         action_space=None,
#         shared_memory=True,
#         copy=True,
#         context=None,
#         daemon=True,
#         worker=None,
#     ):
#         ctx = mp.get_context(context)
#         self.env_fns = env_fns
#         self.shared_memory = shared_memory
#         self.copy = copy
#
#         # Added dummy_env_fn to fix OpenGL error in Mujoco
#         # disable any OpenGL rendering in dummy_env_fn, since it
#         # will conflict with OpenGL context in the forked child process
#         if dummy_env_fn is None:
#             dummy_env_fn = env_fns[0]
#         dummy_env = dummy_env_fn()
#         self.metadata = dummy_env.metadata
#
#         if (observation_space is None) or (action_space is None):
#             observation_space = observation_space or dummy_env.observation_space
#             action_space = action_space or dummy_env.action_space
#         dummy_env.close()
#         del dummy_env
#         super(AsyncVectorEnv, self).__init__(
#             num_envs=len(env_fns),
#             observation_space=observation_space,
#             action_space=action_space,
#         )
#
#         if self.shared_memory:
#             try:
#                 _obs_buffer = create_shared_memory(
#                     self.single_observation_space, n=self.num_envs, ctx=ctx
#                 )
#                 self.observations = read_from_shared_memory(
#                     _obs_buffer, self.single_observation_space, n=self.num_envs
#                 )
#             except CustomSpaceError:
#                 raise ValueError(
#                     "Using `shared_memory=True` in `AsyncVectorEnv` "
#                     "is incompatible with non-standard Gym observation spaces "
#                     "(i.e. custom spaces inheriting from `gym.Space`), and is "
#                     "only compatible with default Gym spaces (e.g. `Box`, "
#                     "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
#                     "if you use custom observation spaces."
#                 )
#         else:
#             _obs_buffer = None
#             self.observations = create_empty_array(
#                 self.single_observation_space, n=self.num_envs, fn=np.zeros
#             )
#
#         self.parent_pipes, self.processes = [], []
#         self.error_queue = ctx.Queue()
#         target = _worker_shared_memory if self.shared_memory else _worker
#         target = worker or target
#         with clear_mpi_env_vars():
#             for idx, env_fn in enumerate(self.env_fns):
#                 parent_pipe, child_pipe = ctx.Pipe()
#                 process = ctx.Process(
#                     target=target,
#                     name="Worker<{0}>-{1}".format(type(self).__name__, idx),
#                     args=(
#                         idx,
#                         CloudpickleWrapper(env_fn),
#                         child_pipe,
#                         parent_pipe,
#                         _obs_buffer,
#                         self.error_queue,
#                     ),
#                 )
#
#                 self.parent_pipes.append(parent_pipe)
#                 self.processes.append(process)
#
#                 process.daemon = daemon
#                 process.start()
#                 child_pipe.close()
#
#         self._state = AsyncState.DEFAULT
#         self._check_observation_spaces()
#
#     def seed(self, seeds=None):
#         self._assert_is_running()
#         if seeds is None:
#             seeds = [None for _ in range(self.num_envs)]
#         if isinstance(seeds, int):
#             seeds = [seeds + i for i in range(self.num_envs)]
#         assert len(seeds) == self.num_envs
#
#         if self._state != AsyncState.DEFAULT:
#             raise AlreadyPendingCallError(
#                 "Calling `seed` while waiting "
#                 "for a pending call to `{0}` to complete.".format(self._state.value),
#                 self._state.value,
#             )
#
#         for pipe, seed in zip(self.parent_pipes, seeds):
#             pipe.send(("seed", seed))
#         _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
#         self._raise_if_errors(successes)
#
#
#     def reset_async(
#         self,
#         seed: Optional[Union[int, List[int]]] = None,
#         options: Optional[dict] = None,
#     ):
#         """Send calls to the :obj:`reset` methods of the sub-environments.
#
#         To get the results of these calls, you may invoke :meth:`reset_wait`.
#
#         Args:
#             seed: List of seeds for each environment
#             options: The reset option
#
#         Raises:
#             ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
#             AlreadyPendingCallError: If the environment is already waiting for a pending call to another
#                 method (e.g. :meth:`step_async`). This can be caused by two consecutive
#                 calls to :meth:`reset_async`, with no call to :meth:`reset_wait` in between.
#         """
#         self._assert_is_running()
#
#         if seed is None:
#             seed = [None for _ in range(self.num_envs)]
#         if isinstance(seed, int):
#             seed = [seed + i for i in range(self.num_envs)]
#         assert len(seed) == self.num_envs
#
#         if self._state != AsyncState.DEFAULT:
#             raise AlreadyPendingCallError(
#                 f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
#                 self._state.value,
#             )
#
#         for pipe, single_seed in zip(self.parent_pipes, seed):
#             single_kwargs = {}
#             if single_seed is not None:
#                 single_kwargs["seed"] = single_seed
#             if options is not None:
#                 single_kwargs["options"] = options
#
#             pipe.send(("reset", single_kwargs))
#         self._state = AsyncState.WAITING_RESET
#
#     def reset_wait(
#         self,
#         timeout: Optional[Union[int, float]] = None,
#         seed: Optional[int] = None,
#         options: Optional[dict] = None,
#     ) -> Union[ObsType, Tuple[ObsType, List[dict]]]:
#         """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.
#
#         Args:
#             timeout: Number of seconds before the call to `reset_wait` times out. If `None`, the call to `reset_wait` never times out.
#             seed: ignored
#             options: ignored
#
#         Returns:
#             A tuple of batched observations and list of dictionaries
#
#         Raises:
#             ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
#             NoAsyncCallError: If :meth:`reset_wait` was called without any prior call to :meth:`reset_async`.
#             TimeoutError: If :meth:`reset_wait` timed out.
#         """
#         self._assert_is_running()
#         if self._state != AsyncState.WAITING_RESET:
#             raise NoAsyncCallError(
#                 "Calling `reset_wait` without any prior " "call to `reset_async`.",
#                 AsyncState.WAITING_RESET.value,
#             )
#
#         if not self._poll(timeout):
#             self._state = AsyncState.DEFAULT
#             raise mp.TimeoutError(
#                 f"The call to `reset_wait` has timed out after {timeout} second(s)."
#             )
#
#         results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
#         self._raise_if_errors(successes)
#         self._state = AsyncState.DEFAULT
#
#         infos = {}
#         results, info_data = zip(*results)
#         for i, info in enumerate(info_data):
#             infos = self._add_info(infos, info, i)
#
#         if not self.shared_memory:
#             self.observations = concatenate(
#                 self.single_observation_space, results, self.observations
#             )
#
#         return (deepcopy(self.observations) if self.copy else self.observations), infos
#
#
#     def step_async(self, actions):
#         """
#         Parameters
#         ----------
#         actions : iterable of samples from `action_space`
#             List of actions.
#         """
#         self._assert_is_running()
#         if self._state != AsyncState.DEFAULT:
#             raise AlreadyPendingCallError(
#                 "Calling `step_async` while waiting "
#                 "for a pending call to `{0}` to complete.".format(self._state.value),
#                 self._state.value,
#             )
#
#         for pipe, action in zip(self.parent_pipes, actions):
#             pipe.send(("step", action))
#         self._state = AsyncState.WAITING_STEP
#
#     def step_wait(self, timeout=None):
#         """
#         Parameters
#         ----------
#         timeout : int or float, optional
#             Number of seconds before the call to `step_wait` times out. If
#             `None`, the call to `step_wait` never times out.
#         Returns
#         -------
#         observations : sample from `observation_space`
#             A batch of observations from the vectorized environment.
#         rewards : `np.ndarray` instance (dtype `np.float_`)
#             A vector of rewards from the vectorized environment.
#         dones : `np.ndarray` instance (dtype `np.bool_`)
#             A vector whose entries indicate whether the episode has ended.
#         infos : list of dict
#             A list of auxiliary diagnostic information.
#         """
#         self._assert_is_running()
#         if self._state != AsyncState.WAITING_STEP:
#             raise NoAsyncCallError(
#                 "Calling `step_wait` without any prior call " "to `step_async`.",
#                 AsyncState.WAITING_STEP.value,
#             )
#
#         if not self._poll(timeout):
#             self._state = AsyncState.DEFAULT
#             raise mp.TimeoutError(
#                 "The call to `step_wait` has timed out after "
#                 "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
#             )
#
#         results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
#         self._raise_if_errors(successes)
#         self._state = AsyncState.DEFAULT
#         observations_list, rewards, dones, infos = zip(*results)
#
#         if not self.shared_memory:
#             self.observations = concatenate(
#                 observations_list, self.observations, self.single_observation_space
#             )
#
#         return (
#             deepcopy(self.observations) if self.copy else self.observations,
#             np.array(rewards),
#             np.array(dones, dtype=np.bool_),
#             infos,
#         )
#
#     def close_extras(self, timeout=None, terminate=False):
#         """
#         Parameters
#         ----------
#         timeout : int or float, optional
#             Number of seconds before the call to `close` times out. If `None`,
#             the call to `close` never times out. If the call to `close` times
#             out, then all processes are terminated.
#         terminate : bool (default: `False`)
#             If `True`, then the `close` operation is forced and all processes
#             are terminated.
#         """
#         timeout = 0 if terminate else timeout
#         try:
#             if self._state != AsyncState.DEFAULT:
#                 logger.warn(
#                     "Calling `close` while waiting for a pending "
#                     "call to `{0}` to complete.".format(self._state.value)
#                 )
#                 function = getattr(self, "{0}_wait".format(self._state.value))
#                 function(timeout)
#         except mp.TimeoutError:
#             terminate = True
#
#         if terminate:
#             for process in self.processes:
#                 if process.is_alive():
#                     process.terminate()
#         else:
#             for pipe in self.parent_pipes:
#                 if (pipe is not None) and (not pipe.closed):
#                     pipe.send(("close", None))
#             for pipe in self.parent_pipes:
#                 if (pipe is not None) and (not pipe.closed):
#                     pipe.recv()
#
#         for pipe in self.parent_pipes:
#             if pipe is not None:
#                 pipe.close()
#         for process in self.processes:
#             process.join()
#
#     def _poll(self, timeout=None):
#         self._assert_is_running()
#         if timeout is None:
#             return True
#         end_time = time.perf_counter() + timeout
#         delta = None
#         for pipe in self.parent_pipes:
#             delta = max(end_time - time.perf_counter(), 0)
#             if pipe is None:
#                 return False
#             if pipe.closed or (not pipe.poll(delta)):
#                 return False
#         return True
#
#     def _check_observation_spaces(self):
#         self._assert_is_running()
#         for pipe in self.parent_pipes:
#             pipe.send(("_check_observation_space", self.single_observation_space))
#         same_spaces, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
#         self._raise_if_errors(successes)
#         if not all(same_spaces):
#             raise RuntimeError(
#                 "Some environments have an observation space "
#                 "different from `{0}`. In order to batch observations, the "
#                 "observation spaces from all environments must be "
#                 "equal.".format(self.single_observation_space)
#             )
#
#     def _assert_is_running(self):
#         if self.closed:
#             raise ClosedEnvironmentError(
#                 "Trying to operate on `{0}`, after a "
#                 "call to `close()`.".format(type(self).__name__)
#             )
#
#     def _raise_if_errors(self, successes):
#         if all(successes):
#             return
#
#         num_errors = self.num_envs - sum(successes)
#         assert num_errors > 0
#         for _ in range(num_errors):
#             index, exctype, value = self.error_queue.get()
#             logger.error(
#                 "Received the following error from Worker-{0}: "
#                 "{1}: {2}".format(index, exctype.__name__, value)
#             )
#             logger.error("Shutting down Worker-{0}.".format(index))
#             self.parent_pipes[index].close()
#             self.parent_pipes[index] = None
#
#         logger.error("Raising the last exception back to the main process.")
#         raise exctype(value)
#     
#     def call_async(self, name: str, *args, **kwargs):
#         """Calls the method with name asynchronously and apply args and kwargs to the method.
#
#         Args:
#             name: Name of the method or property to call.
#             *args: Arguments to apply to the method call.
#             **kwargs: Keyword arguments to apply to the method call.
#
#         Raises:
#             ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
#             AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
#         """
#         self._assert_is_running()
#         if self._state != AsyncState.DEFAULT:
#             raise AlreadyPendingCallError(
#                 "Calling `call_async` while waiting "
#                 f"for a pending call to `{self._state.value}` to complete.",
#                 self._state.value,
#             )
#
#         for pipe in self.parent_pipes:
#             pipe.send(("_call", (name, args, kwargs)))
#         self._state = AsyncState.WAITING_CALL
#
#     def call_wait(self, timeout = None) -> list:
#         """Calls all parent pipes and waits for the results.
#
#         Args:
#             timeout: Number of seconds before the call to `step_wait` times out.
#                 If `None` (default), the call to `step_wait` never times out.
#
#         Returns:
#             List of the results of the individual calls to the method or property for each environment.
#
#         Raises:
#             NoAsyncCallError: Calling `call_wait` without any prior call to `call_async`.
#             TimeoutError: The call to `call_wait` has timed out after timeout second(s).
#         """
#         self._assert_is_running()
#         if self._state != AsyncState.WAITING_CALL:
#             raise NoAsyncCallError(
#                 "Calling `call_wait` without any prior call to `call_async`.",
#                 AsyncState.WAITING_CALL.value,
#             )
#
#         if not self._poll(timeout):
#             self._state = AsyncState.DEFAULT
#             raise mp.TimeoutError(
#                 f"The call to `call_wait` has timed out after {timeout} second(s)."
#             )
#
#         results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
#         self._raise_if_errors(successes)
#         self._state = AsyncState.DEFAULT
#
#         return results
#
#     def call(self, name: str, *args, **kwargs):
#         """Call a method, or get a property, from each parallel environment.
#
#         Args:
#             name (str): Name of the method or property to call.
#             *args: Arguments to apply to the method call.
#             **kwargs: Keyword arguments to apply to the method call.
#
#         Returns:
#             List of the results of the individual calls to the method or property for each environment.
#         """
#         self.call_async(name, *args, **kwargs)
#         return self.call_wait()
#     
#
#     def call_each(self, name: str, 
#             args_list: list=None, 
#             kwargs_list: list=None, 
#             timeout = None):
#         n_envs = len(self.parent_pipes)
#         if args_list is None:
#             args_list = [[]] * n_envs
#         assert len(args_list) == n_envs
#
#         if kwargs_list is None:
#             kwargs_list = [dict()] * n_envs
#         assert len(kwargs_list) == n_envs
#
#         # send
#         self._assert_is_running()
#         if self._state != AsyncState.DEFAULT:
#             raise AlreadyPendingCallError(
#                 "Calling `call_async` while waiting "
#                 f"for a pending call to `{self._state.value}` to complete.",
#                 self._state.value,
#             )
#
#         for i, pipe in enumerate(self.parent_pipes):
#             pipe.send(("_call", (name, args_list[i], kwargs_list[i])))
#         self._state = AsyncState.WAITING_CALL
#
#         # receive
#         self._assert_is_running()
#         if self._state != AsyncState.WAITING_CALL:
#             raise NoAsyncCallError(
#                 "Calling `call_wait` without any prior call to `call_async`.",
#                 AsyncState.WAITING_CALL.value,
#             )
#
#         if not self._poll(timeout):
#             self._state = AsyncState.DEFAULT
#             raise mp.TimeoutError(
#                 f"The call to `call_wait` has timed out after {timeout} second(s)."
#             )
#
#         results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
#         self._raise_if_errors(successes)
#         self._state = AsyncState.DEFAULT
#
#         return results
#
#
#     def set_attr(self, name: str, values):
#         """Sets an attribute of the sub-environments.
#
#         Args:
#             name: Name of the property to be set in each individual environment.
#             values: Values of the property to be set to. If ``values`` is a list or
#                 tuple, then it corresponds to the values for each individual
#                 environment, otherwise a single value is set for all environments.
#
#         Raises:
#             ValueError: Values must be a list or tuple with length equal to the number of environments.
#             AlreadyPendingCallError: Calling `set_attr` while waiting for a pending call to complete.
#         """
#         self._assert_is_running()
#         if not isinstance(values, (list, tuple)):
#             values = [values for _ in range(self.num_envs)]
#         if len(values) != self.num_envs:
#             raise ValueError(
#                 "Values must be a list or tuple with length equal to the "
#                 f"number of environments. Got `{len(values)}` values for "
#                 f"{self.num_envs} environments."
#             )
#
#         if self._state != AsyncState.DEFAULT:
#             raise AlreadyPendingCallError(
#                 "Calling `set_attr` while waiting "
#                 f"for a pending call to `{self._state.value}` to complete.",
#                 self._state.value,
#             )
#
#         for pipe, value in zip(self.parent_pipes, values):
#             pipe.send(("_setattr", (name, value)))
#         _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
#         self._raise_if_errors(successes)
#
#     def render(self, *args, **kwargs):
#         return self.call('render', *args, **kwargs)
#
#
#
# def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
#     assert shared_memory is None
#     env = env_fn()
#     parent_pipe.close()
#     try:
#         while True:
#             command, data = pipe.recv()
#             if command == "reset":
#                 observation = env.reset()
#                 pipe.send((observation, True))
#             elif command == "step":
#                 observation, reward, done, info = env.step(data)
#                 # if done:
#                 #     observation = env.reset()
#                 pipe.send(((observation, reward, done, info), True))
#             elif command == "seed":
#                 env.seed(data)
#                 pipe.send((None, True))
#             elif command == "close":
#                 pipe.send((None, True))
#                 break
#             elif command == "_call":
#                 name, args, kwargs = data
#                 if name in ["reset", "step", "seed", "close"]:
#                     raise ValueError(
#                         f"Trying to call function `{name}` with "
#                         f"`_call`. Use `{name}` directly instead."
#                     )
#                 function = getattr(env, name)
#                 if callable(function):
#                     pipe.send((function(*args, **kwargs), True))
#                 else:
#                     pipe.send((function, True))
#             elif command == "_setattr":
#                 name, value = data
#                 setattr(env, name, value)
#                 pipe.send((None, True))
#
#             elif command == "_check_observation_space":
#                 pipe.send((data == env.observation_space, True))
#             else:
#                 raise RuntimeError(
#                     "Received unknown command `{0}`. Must "
#                     "be one of {`reset`, `step`, `seed`, `close`, "
#                     "`_check_observation_space`}.".format(command)
#                 )
#     except (KeyboardInterrupt, Exception) as e:
#         error_queue.put((index, type(e), traceback.format_exc()))
#         pipe.send((None, False))
#     finally:
#         env.close()
#
#
# def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
#     assert shared_memory is not None
#     env = env_fn()
#     observation_space = env.observation_space
#     parent_pipe.close()
#     try:
#         while True:
#             command, data = pipe.recv()
#             if command == "reset":
#                 observation = env.reset()
#                 write_to_shared_memory(
#                     index, observation, shared_memory, observation_space
#                 )
#                 pipe.send((None, True))
#             elif command == "step":
#                 observation, reward, done, info = env.step(data)
#                 # if done:
#                 #     observation = env.reset()
#                 write_to_shared_memory(
#                     index, observation, shared_memory, observation_space
#                 )
#                 pipe.send(((None, reward, done, info), True))
#             elif command == "seed":
#                 env.seed(data)
#                 pipe.send((None, True))
#             elif command == "close":
#                 pipe.send((None, True))
#                 break
#             elif command == "_call":
#                 name, args, kwargs = data
#                 if name in ["reset", "step", "seed", "close"]:
#                     raise ValueError(
#                         f"Trying to call function `{name}` with "
#                         f"`_call`. Use `{name}` directly instead."
#                     )
#                 function = getattr(env, name)
#                 if callable(function):
#                     pipe.send((function(*args, **kwargs), True))
#                 else:
#                     pipe.send((function, True))
#             elif command == "_setattr":
#                 name, value = data
#                 setattr(env, name, value)
#                 pipe.send((None, True))
#             elif command == "_check_observation_space":
#                 pipe.send((data == observation_space, True))
#             else:
#                 raise RuntimeError(
#                     "Received unknown command `{0}`. Must "
#                     "be one of {`reset`, `step`, `seed`, `close`, "
#                     "`_check_observation_space`}.".format(command)
#                 )
#     except (KeyboardInterrupt, Exception) as e:
#         # error_queue.put((index,) + sys.exc_info()[:2])
#         # error_queue.put((index, type(e), e))
#         error_queue.put((index, type(e), traceback.format_exc()))
#         pipe.send((None, False))
#     finally:
#         env.close()
