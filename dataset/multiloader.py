
from multiprocessing import cpu_count
from typing import Any, Callable, Literal, Mapping, Sequence, Type, Union
import numpy as np
import torch
from ffcv.fields.base import Field
from ffcv.loader import Loader
from ffcv.loader.loader import (DEFAULT_PROCESS_CACHE, ORDER_MAP, ORDER_TYPE,
                                OrderOption)
from ffcv.memory_managers import (MemoryManager, OSCacheManager,
                                  ProcessCacheManager)
from ffcv.pipeline import Compiler, Pipeline, PipelineSpec
from ffcv.pipeline.graph import Graph
from ffcv.pipeline.operation import Operation
from ffcv.reader import Reader
from ffcv.traversal_order.base import TraversalOrder


class MultiLoader(Loader):
    def __init__(self,
                fname: str,
                batch_size: int,
                num_workers: int = -1,
                cache_type: int = DEFAULT_PROCESS_CACHE,
                order: Union[ORDER_TYPE, TraversalOrder] = OrderOption.SEQUENTIAL,
                distributed: bool = False,
                seed: int = None,  # For ordering of samples
                indices: Sequence[int] = None,  # For subset selection
                pipelines: Mapping[str,
                                Sequence[Union[Operation, torch.nn.Module]]] = {},
                custom_fields: Mapping[str, Type[Field]] = {},
                drop_last: bool = True,
                batches_ahead: int = 3,
                recompile: bool = False,  # Recompile at every epoch
                ):

        if distributed and order == OrderOption.RANDOM and (seed is None):
            print('Warning: no ordering seed was specified with distributed=True. '
                    'Setting seed to 0 to match PyTorch distributed sampler.')
            seed = 0
        elif seed is None:
            tinfo = np.iinfo('int32')
            seed = np.random.randint(0, tinfo.max)

        # We store the original user arguments to be able to pass it to the
        # filtered version of the datasets
        self._args = {
            'fname': fname,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'os_cache': cache_type,
            'order': order,
            'distributed': distributed,
            'seed': seed,
            'indices': indices,
            'pipelines': pipelines,
            'drop_last': drop_last,
            'batches_ahead': batches_ahead,
            'recompile': recompile
        }
        self.fname: str = fname
        self.batch_size: int = batch_size
        self.batches_ahead = batches_ahead
        self.seed: int = seed
        self.reader: Reader = Reader(self.fname, custom_fields)
        self.num_workers: int = num_workers
        self.drop_last: bool = drop_last
        self.distributed: bool = distributed
        self.code = None
        self.recompile = recompile

        if self.num_workers < 1:
            self.num_workers = cpu_count()

        Compiler.set_num_threads(self.num_workers)

        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)

        
        if cache_type == 0:
            self.memory_manager: MemoryManager = OSCacheManager(self.reader)
        elif cache_type == 1:
            self.memory_manager: MemoryManager = ProcessCacheManager(
                self.reader)
        elif cache_type == 2:
            from ffcv.memory_managers.shared_cache import SharedMemoryManager
            self.memory_manager: MemoryManager = SharedMemoryManager(self.reader)
        else:
            raise ValueError("Unknown cache type. Use 0 for process cache, 1 for os cache, or 2 for no cache.")
        
        if order in ORDER_MAP:
            self.traversal_order: TraversalOrder = ORDER_MAP[order](self)
        elif isinstance(order, TraversalOrder):
            self.traversal_order: TraversalOrder = order(self)
        elif issubclass(order, TraversalOrder):
            self.traversal_order: TraversalOrder = order(self)
        else:
            raise ValueError(f"Order {order} is not a supported order type or a subclass of TraversalOrder")

        memory_read = self.memory_manager.compile_reader()
        self.next_epoch: int = 0

        self.pipelines = {}
        self.pipeline_specs = {}
        self.field_name_to_f_ix = {}
        custom_pipeline_specs = {}

        # Creating PipelineSpec objects from the pipeline dict passed
        # by the user
        for output_name, spec in pipelines.items():
            if isinstance(spec, PipelineSpec):
                pass
            elif isinstance(spec, Sequence):
                spec = PipelineSpec(output_name, decoder=None, transforms=spec)
            elif spec is None:
                continue  # This is a disabled field
            else:
                msg  = f"The pipeline for {output_name} has to be "
                msg += f"either a PipelineSpec or a sequence of operations"
                raise ValueError(msg)
            custom_pipeline_specs[output_name] = spec

        # Adding the default pipelines
        default_name_to_f_ix={}
        for f_ix, (field_name, field) in enumerate(self.reader.handlers.items()):
            default_name_to_f_ix[field_name] = f_ix

        # We add the custom fields after the default ones
        # This is to preserve backwards compatibility and make sure the order
        # is intuitive
        for field_name, spec in custom_pipeline_specs.items():
            # redirect
            self.field_name_to_f_ix[field_name] = default_name_to_f_ix[spec.source]

            if field_name not in self.pipeline_specs:
                self.pipeline_specs[field_name] = spec

        self.graph = Graph(self.pipeline_specs, self.reader.handlers,
                            self.field_name_to_f_ix, self.reader.metadata,
                            memory_read)
        
        self.generate_code()
        self.first_traversal_order = self.next_traversal_order()