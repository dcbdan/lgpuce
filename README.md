# The Little ((GPU Node) (Command Executor))

### TODO

* define the commandset here $\frac{a}{b}$
* create execution engine
* create example gpu kernels
* create example graphs

## Building

The dependencies are cudnn and a C++ compiler.

To build with singularity:

```
# build the container (this requires the cudnn package)
singularity build --fakeroot container.sif container.def

# build the executable
./compile

# run
singularity exec --nv container.sif ./exp
```
