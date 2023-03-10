# The Little ((GPU Node) (Command Executor))

Execute a command graph on a GPU node.
A command graph is a compute graph with
`apply` and `sendrecv` commands as vertices
and dependencies as edges.

An apply command says 'Execute this op, with
this read memory, this write memory at this location',
where a location is either a CPU device or GPU device.
(Here, a Lgpunce CPU really just means the host, and can use multiple
physical CPUs--a Lgpunce GPU really is just referring to a physical GPU.
For example, in the future, we could have each Lgpunce CPU device
really refer to a single NUMA node.)

A sendrecv command represents point to point communication between
locations.

All commands can have dependencies, forming the edges of the
command graph. An edge is only valid if the child and parent
commands happen on the same graph. Sending from gpu 0 to gpu 1
cannot depend on sending from gpu 1 to gpu 2.

For any memory that is being written to, only one command
can be executed at a time, regardless of dependencies.

## Building

The dependencies are Cuda, Cudnn and a C++ compiler.

To build with [Apptainer](apptainer.org):

```
# build the container (this requires the cudnn package)
singularity build --fakeroot container.sif container.def

# build the executable
./compile # singularity exec <nvcc stuff>

# run and collect nvvp profile stats profile
# (generated .nvvp can be opened in nvidia visual profiler)
./run none                            <args>
./run timeline run3_timeline.nvvp     <args>
./run nvlink   run3_nvlink_stats.nvvp <args>
```


