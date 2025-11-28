# ECDLP-LasVegas
## Parallel Implementation of the LasVegas algorithm

### Las Vegas : High Level algorithm
1. Generate Weight Vector
2. Generate unique random number (in sequence) for each processor
3. Use random numbers to compute a matrix **M**
4. Compute left kernel of **M**
5. Reduce **M** such that a row has *r-zeros*
6. If such a row exists then STOP else goto step 2.