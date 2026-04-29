from __future__ import annotations
import multiprocessing as mp
import numpy as np

def _worker(remote, env_cls, kwargs):
    env=env_cls(**kwargs)
    while True:
        cmd,data=remote.recv()
        if cmd=='reset': remote.send(env.reset())
        elif cmd=='step': remote.send(env.step(data))
        elif cmd=='close': remote.close(); break
class ParallelEnvRunner:
    def __init__(self, env_cls, env_kwargs_list):
        self.closed=False; self.remotes,self.work_remotes=zip(*[mp.Pipe() for _ in env_kwargs_list]); self.ps=[mp.Process(target=_worker,args=(wr,env_cls,kwargs),daemon=True) for wr,kwargs in zip(self.work_remotes,env_kwargs_list)]
        for p in self.ps: p.start()
        for wr in self.work_remotes: wr.close()
        self.num_envs=len(env_kwargs_list)
    def reset(self):
        for remote in self.remotes: remote.send(('reset',None))
        obs_list=[remote.recv() for remote in self.remotes]
        return {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0].keys()}
    def step(self, actions):
        for remote,action in zip(self.remotes,actions): remote.send(('step',action))
        results=[remote.recv() for remote in self.remotes]
        obs_list,rewards,dones,infos=zip(*results)
        return {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0].keys()}, np.asarray(rewards,dtype=np.float32), np.asarray(dones,dtype=np.bool_), list(infos)
    def close(self):
        if self.closed: return
        for remote in self.remotes: remote.send(('close',None))
        for p in self.ps: p.join()
        self.closed=True
