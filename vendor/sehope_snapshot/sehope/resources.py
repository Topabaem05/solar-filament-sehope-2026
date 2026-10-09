"""Bounded MPS allocation with system-wide host headroom checks."""
from dataclasses import dataclass
import math
import os
import psutil
import torch

GIB = 1024**3


@dataclass
class MemoryBudget:
    limit_bytes: int
    reserve_bytes: int
    max_swap_growth_bytes: int
    baseline_swap_bytes: int
    sampled_peak_driver_bytes: int = 0
    sampled_peak_allocated_bytes: int = 0
    samples: int = 0

    def __post_init__(self):
        if self.limit_bytes <= 0 or min(self.reserve_bytes, self.max_swap_growth_bytes,
                                       self.baseline_swap_bytes) < 0:
            raise ValueError('Invalid memory budget')

    def check_start(self, available, recommended):
        if self.limit_bytes > recommended:
            raise RuntimeError('MPS budget exceeds device recommended maximum')
        if available < self.limit_bytes + self.reserve_bytes:
            raise RuntimeError('Insufficient host headroom for MPS budget plus reserve; wait')

    def check_sample(self, available, swap, driver, allocated):
        if available < self.reserve_bytes:
            raise RuntimeError('Host memory reserve breached; stop this SE-HOPE run')
        if swap - self.baseline_swap_bytes > self.max_swap_growth_bytes:
            raise RuntimeError('System-wide swap growth exceeded budget; stop this SE-HOPE run')
        if max(driver, allocated) > self.limit_bytes:
            raise RuntimeError('MPS process memory budget exceeded; stop this SE-HOPE run')

    def sample(self, phase):
        torch.mps.synchronize()
        available, swap = psutil.virtual_memory().available, psutil.swap_memory().used
        driver, allocated = torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory()
        self.sampled_peak_driver_bytes = max(self.sampled_peak_driver_bytes, driver)
        self.sampled_peak_allocated_bytes = max(self.sampled_peak_allocated_bytes, allocated)
        self.samples += 1
        self.check_sample(available, swap, driver, allocated)
        return {'phase': phase, 'driver_bytes': driver, 'allocated_bytes': allocated,
                'sampled_peak_driver_bytes': self.sampled_peak_driver_bytes,
                'sampled_peak_allocated_bytes': self.sampled_peak_allocated_bytes,
                'host_available_bytes': available, 'host_swap_bytes': swap,
                'baseline_swap_bytes': self.baseline_swap_bytes, 'samples': self.samples,
                'mps_limit_bytes': self.limit_bytes, 'host_reserve_bytes': self.reserve_bytes,
                'max_swap_growth_bytes': self.max_swap_growth_bytes,
                'scope': 'Synchronized phase-boundary samples; not exact peak. Host metrics include other processes.'}


def configure_mps(config):
    """Call before placing model tensors on MPS. Never enable CPU fallback."""
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Disable PYTORCH_ENABLE_MPS_FALLBACK for verified MPS execution')
    version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if version < (2, 6):
        raise RuntimeError('Use requirements-mps.txt: PyTorch 2.5.1 fails suffix convolution backward on MPS')
    runtime = config.get('runtime', {})
    def byte_value(key, default):
        value = float(runtime.get(key, default))
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite memory setting: {key}')
        return int(value * GIB)
    budget = MemoryBudget(byte_value('mps_memory_gib', 2), byte_value('host_reserve_gib', 4),
                          byte_value('max_swap_growth_gib', .5), psutil.swap_memory().used)
    recommended = torch.mps.recommended_max_memory()
    budget.check_start(psutil.virtual_memory().available, recommended)
    torch.mps.set_per_process_memory_fraction(budget.limit_bytes / recommended)
    return budget

@dataclass
class HostMemoryBudget:
    reserve_bytes: int
    max_swap_growth_bytes: int
    baseline_swap_bytes: int
    sampled_peak_rss_bytes: int = 0
    samples: int = 0

    def sample(self,phase):
        available=psutil.virtual_memory().available;swap=psutil.swap_memory().used
        rss=psutil.Process().memory_info().rss
        self.samples+=1;self.sampled_peak_rss_bytes=max(self.sampled_peak_rss_bytes,rss)
        if available<self.reserve_bytes:raise RuntimeError('Host reserve breached during CPU prefix processing')
        if swap-self.baseline_swap_bytes>self.max_swap_growth_bytes:raise RuntimeError('Swap growth exceeded CPU prefix budget')
        return dict(phase=phase,host_available_bytes=available,host_swap_bytes=swap,
                    baseline_swap_bytes=self.baseline_swap_bytes,rss_bytes=rss,
                    sampled_peak_rss_bytes=self.sampled_peak_rss_bytes,samples=self.samples,
                    scope='CPU process RSS and system-wide host metrics; synchronized batch boundaries, not exact peak.')


def configure_host(config):
    runtime=config.get('runtime',{})
    guard=HostMemoryBudget(int(runtime.get('host_reserve_gib',4)*GIB),
                           int(runtime.get('max_swap_growth_gib',.5)*GIB),psutil.swap_memory().used)
    guard.sample('before_cpu_model')
    return guard

@dataclass
class CudaMemoryBudget:
    limit_bytes: int
    reserve_bytes: int
    max_swap_growth_bytes: int
    baseline_swap_bytes: int
    samples: int = 0

    def __post_init__(self):
        if self.limit_bytes<=0 or min(self.reserve_bytes,self.max_swap_growth_bytes,self.baseline_swap_bytes)<0:
            raise ValueError('Invalid CUDA budget')

    def check_start(self,available,total,free):
        if available<self.reserve_bytes:raise RuntimeError('Insufficient host reserve for CUDA run')
        if self.limit_bytes>total or free<self.limit_bytes+512*2**20:
            raise RuntimeError('Insufficient CUDA capacity for allocator cap plus context reserve')

    def check_sample(self,available,swap,reserved,allocated):
        if available<self.reserve_bytes:raise RuntimeError('CUDA run breached host reserve')
        if swap-self.baseline_swap_bytes>self.max_swap_growth_bytes:raise RuntimeError('CUDA run exceeded system swap-growth budget')
        if max(reserved,allocated)>self.limit_bytes:raise RuntimeError('CUDA allocator budget exceeded')

    def sample(self,phase):
        torch.cuda.synchronize()
        available=psutil.virtual_memory().available;swap=psutil.swap_memory().used
        reserved=torch.cuda.memory_reserved();allocated=torch.cuda.memory_allocated()
        self.check_sample(available,swap,reserved,allocated);self.samples+=1
        return dict(phase=phase,cuda_allocated_bytes=allocated,cuda_reserved_bytes=reserved,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),cuda_limit_bytes=self.limit_bytes,
                    host_available_bytes=available,host_swap_bytes=swap,baseline_swap_bytes=self.baseline_swap_bytes,
                    samples=self.samples,scope='PyTorch CUDA allocator peaks exclude CUDA context and non-PyTorch allocations.')


def configure_cuda(config):
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    runtime=config.get('runtime',{})
    def amount(key,default):
        value=float(runtime.get(key,default))
        if not math.isfinite(value):raise ValueError('Nonfinite CUDA budget')
        return int(value*GIB)
    budget=CudaMemoryBudget(amount('cuda_memory_gib',4),amount('host_reserve_gib',4),
                            amount('max_swap_growth_gib',.5),psutil.swap_memory().used)
    free,total=torch.cuda.mem_get_info()
    budget.check_start(psutil.virtual_memory().available,total,free)
    torch.cuda.set_per_process_memory_fraction(budget.limit_bytes/total)
    torch.cuda.reset_peak_memory_stats()
    # Comparable FP32 arithmetic on devices that otherwise permit TF32.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return budget
