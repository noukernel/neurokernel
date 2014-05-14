Max Miller, Patrick Clare, and Nicole Rivilis
Neuro Project Readme
5/14/14

This zip file contains the host of files we used to run our simulation of the fly’s visual system.  For a more extensive file base, please look at our Github www.github.com/noukernel/neurokernel

MATLAB Files:
neurokernel/misc/Matlab_stuff/

Signal_Cascade.m: This is the function that takes in an X vector and updates it for a single randomly determined dt.  Uses the Song differential equations

F_HHN.m: The function that simulates the modified Hodgkin-Huxley neuron.

Photoreceptor_run.m: This is the executable file of the MATLAB section.  Establishes parameters, instantiates vectors, and runs the signal cascade for a single microvillus using randomly determined inputs.

Python Files:
neurokernel/neurokernel/LPU/neurons/photoreceptor.py:
This is the photoreceptor model. Has three major kernels: signal cascade, calcium dynamics, and the modified HH neuron.  Also has a few other minor kernels for random number generation for photon absorption.  We chose to run 512 threads per block (any more and we ran out of memory because of other computational needs).  This way, we could use the neuron number as the block index.  The calcium dynamics kernel also sums the microvillus currents for each photoreceptor for input into the HH neuron, using __syncthreads().  The outer Python code copies everything from the CPU to the GPU and then back again, calling each kernel.  

neurokernel/misc/retinatest.py:
This Python function was written to take the input h5 file of the image and generate an h5 file of neuron outputs, using photoreceptor.py. We define a time step (dt) of 1e-4 and a total duration of 1.0 seconds. We create nodes based on the photoreceptor non-spiking neuron model with an initial voltage of -70mV. 





