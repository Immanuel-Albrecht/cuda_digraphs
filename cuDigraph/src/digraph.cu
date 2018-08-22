#include "pch.h"

#include "digraph.h"

namespace cuDigraph {

	/* cuDigraph helper kernels */

	__global__ void dev_augment_paths(
		dev_family_of_paths F,  // input&output paths
		dev_incidence_record I, // in-neighbor incidence structure
		bool *batchOverflow,    // raise a flag in case of overflow of output batch
		family_size n0          // in case we have to call multiple kernels on the data set...
	) {
		auto n = blockIdx.x * blockDim.x + threadIdx.x + n0;
		if (n < F.nInput) {
			auto &p = F.dev_input[n]; // input path
			auto v1 = p.v[0]; // first vertex of the path
			auto Iv1 = I.incidence + I.start[v1]; // start of in-neighbor incidence list
			int m = 0; // output path number
			for (auto idx = 0; idx < I.length[v1]; ++idx) {
				auto v0 = Iv1[idx];
				if (!is_in_vertex_set(p.support, v0)) 
				{
					if (m == F.mOutput)
					{
						if (batchOverflow) /* if zero, do not give feedback. */
							*batchOverflow = true;
					}
					else {
						// prepend v0 to p and output new path
						auto &q = F.dev_output[n*F.mOutput + m];

						q.n = p.n + 1;

						copy_vertex_set(p.support, q.support);
						set_in_vertex_set(q.support, v0);

						q.v[0] = v0;
						for (auto i = 0; i < p.n; ++i)
							q.v[i + 1] = p.v[i];
						

						++m;
					}
				}
			}
		}
	}


	struct predicate_path_nonzero {
		__host__ __device__
			bool operator()(const dev_walk& x)
		{
			return x.n != 0;
		}
	};

	void compact_family_of_paths_output_to_input(dev_family_of_paths& devF) {

		assert(devF.resInput >= devF.nInput*devF.mOutput);


		thrust::device_ptr<dev_walk> sparse = thrust::device_pointer_cast(devF.dev_output);	
		thrust::device_ptr<dev_walk> compact = thrust::device_pointer_cast(devF.dev_input);


		auto end = thrust::copy_if(sparse, sparse + devF.nInput*devF.mOutput, compact, predicate_path_nonzero());

		devF.nInput = (family_size) (end - compact);

	}

	void augment_paths(int blocks, int threads,
		dev_family_of_paths& F,  // input&output paths
		dev_incidence_record& I, // in-neighbor incidence structure
		bool *batchOverflow,    // raise a flag in case of overflow of output batch
		family_size n0          // in case we have to call multiple kernels on the data set...
	) {
		cudaKernel(dev_augment_paths, blocks, threads)(F, I, batchOverflow, n0);
	}

}