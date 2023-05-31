# parallel-matrix-multiplication-opencl

### Matrix Multiplication with OpenCL
Implementation of an OpenCL matrix multiplication application with the following independent variables:

1.  The width of the matrices
2.  The width of the grid of threads per block (work items per workgroup)

### Commentary
1. Machine/environment:
	- Machine used: `Rabbit + DGX`
	- Operating system used: `Linux`
	- The compiler used: `g++`
2. Experiment visualizations
	- Table of total matrix size and total local size vs performance values![[styled_table.svg]]
	- Graph 1 Performance Vs. Total Matrix Size![performance_total_size.svg](https://github.com/ztbochanski/parallel-matrix-multiplication-opencl/blob/2d6d2c26c318028fd0aaa8c6e8a20a8527503ff5/performance_total_size.svg)
	- Graph 2 Performance. Vs. Total Local. Size![[performance_local_size.svg]]
3. What patterns are you seeing in the performance curves? What difference does the size of the matrices make? What difference does the size of each work group make?
	- Patterns: Larger matrix sizes do better in general than smaller sizes. As workgroup sizes increase performance increases. Both have diminishing returns as local size and total matrix size increase.
	- Matrix Size: Increasing matrix size indicates an increase in performance initially followed by quickly diminishing returns.
	- Local Size: Varying the work-group size allows a look into how the parallel performance is impacted. It appears increasing the workgroup size indicates an improvement in performance, then diminishing returns, and finally a decrease in performance. This shows an optimal workgroup size, which probably depends on the hardware and other device characteristics.
4. Why do you think the patterns look this way?
	- The increase in performance with the larger matrices is likely due to parallelism As size increases more parallel computation can be performed simultaneously. This is shown again in workgroup size vs performance and tells us that larger workgroups mean more work elements can be executed concurrently.
	- Since there is a decrease in performance eventually, this shows that device resources might be the limiting factor because when the device is maxed out increasing the matrix size shows diminishing returns and a plateau. This pattern is reflected again in increasing workgroup size. The diminishing performance is most likely due to architecture bottlenecks like memory access or the like.
	- Ultimately, as both matrix and workgroup size increase, there is a performance increase until there is an optimal combination of matrix and workgroup size, before a decrease in performance occurs.
	- It's important to note that there might be some specific details of this experiment that would look different on different hardware, or a different language (algorithmic implementations under the hood of OpenCL) but overall the patterns would hold.
