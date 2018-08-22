#include "pch.h"

#include "digraph.h"

void cuDigraph::dev_incidence_record_::unload()
{
	cudaSafeCall(    cudaFree(this->start));
	cudaSafeCall(cudaFree(this->incidence));
	cudaSafeCall(   cudaFree(this->length));
	this->nV = 0;
	this->incidence = 0;
	this->length = 0;
	this->start = 0;
}

cuDigraph::dev_incidence_record cuDigraph::Digraph::load() const
{
	/* this could be a function of incidence_record, too; 
	but since we use it only once, its a member of cuDigraph.*/

	dev_incidence_record devI;

	auto I = encode();
	
	assert(I.isValid());
	
	
	devI.nV = (vertex) I.start.size();
	devI.maxInDegree = maxInDegree();
	

	cudaSafeMallocVector(devI.start, I.start.size());
	cudaSafeMallocVector(devI.length, I.length.size());
	cudaSafeMallocVector(devI.incidence, I.incidence.size());
	
	cudaSafeVectorToDev(devI.start, I.start.data(), I.start.size());
	cudaSafeVectorToDev(devI.length, I.length.data(), I.length.size());
	cudaSafeVectorToDev(devI.incidence, I.incidence.data(), I.incidence.size());

	return devI;
}

cuDigraph::FamilyOfPathsSTL cuDigraph::Digraph::stl_allPaths(int max_len, int blocks, int threads) const
{
	return stl_allPathsTo(V.begin(), V.end(), max_len, blocks, threads);
}

cuDigraph::FamilyOfPaths cuDigraph::Digraph::allPaths(int max_len, int initial_allocation_size, int maximal_device_size, int blocks, int threads) const
{
	return allPathsTo(V.begin(), V.end(), max_len,  initial_allocation_size, maximal_device_size, blocks, threads);
}

void cuDigraph::dev_family_of_paths_::enlargeOutput()
{
	if (resOutput >= nInput * mOutput)
		return; /* already large enough. */

	auto old_output = dev_output;

	resOutput = nInput*mOutput;

	cudaSafeMallocVector(dev_output, resOutput);

	cudaSafeCall(cudaFree(old_output));
}

void cuDigraph::dev_family_of_paths_::enlargeInput()
{
	if (resInput >= nInput*mOutput)
		return; /* large enough */

	auto old_input = dev_input;

	resInput = nInput * mOutput;

	cudaSafeMallocVector(dev_input, resInput);

	cudaSafeDeviceVectorMemcpy(dev_input, old_input, nInput);

	cudaSafeCall(cudaFree(old_input));
}

void cuDigraph::dev_family_of_paths_::enlargeToLimit(family_size limit)
{
	if ((resInput == limit) && (resOutput == limit))
		return;

	cudaSafeCall(cudaFree(dev_input));
	cudaSafeCall(cudaFree(dev_output));

	cudaSafeMallocVector(dev_input, limit);
	cudaSafeMallocVector(dev_output, limit);

	resInput = limit;
	resOutput = limit;
	nInput = 0;
}

void cuDigraph::dev_family_of_paths_::unload()
{
	cudaSafeCall( cudaFree(this->dev_input));
	cudaSafeCall(cudaFree(this->dev_output));
}

void cuDigraph::dev_family_of_paths_::clearOutput()
{
	cudaSafeCall(  cudaMemset2D(this->dev_output, sizeof(this->dev_output[0]), 0, sizeof(this->dev_output[0].n), this->mOutput * this->nInput));
}

void cuDigraph::dev_family_of_paths_::clearInput()
{
	cudaSafeCall(cudaMemset2D(this->dev_input, sizeof(this->dev_output[0]), 0, sizeof(this->dev_output[0].n), this->nInput));
}


cuDigraph::dev_family_of_paths cuDigraph::FamilyOfPathsSTL::load(family_size mOutput, family_size allocOutput,
	family_size allocInput, bool symmetric)
{
	assert(mOutput > 0);
	

	dev_family_of_paths devF;

	allocOutput = std::max<family_size>({allocOutput,(family_size) R.size() * mOutput, allocInput*mOutput });

	if (symmetric) {
		allocInput = std::max<family_size>( {allocInput, allocOutput });
		allocOutput = allocInput;
	}
	else {
		allocInput = std::max<family_size>({ allocInput,(family_size) R.size() });
	}

	cudaSafeMallocVector(devF.dev_input, allocInput);
	cudaSafeMallocVector(devF.dev_output, allocOutput);
	devF.resInput = allocInput;
	devF.resOutput = allocOutput;
	devF.mOutput = mOutput;

	devF.nInput = (family_size)R.size();
	devF.clearOutput();
	

	std::vector<dev_walk> input_paths;
	input_paths.reserve(R.size());

	for (auto p : R) {
		assert(p.v.size() < max_path_length);

		dev_walk dp;
		dp.n = (path_size) p.v.size();
		zero_out_vertex_set(dp.support);

		for (auto i = 0; i < dp.n; ++i) {
			assert(p.v[i] < max_vertex_set_vertex_id);

			dp.v[i] = p.v[i];
			set_in_vertex_set(dp.support, p.v[i]);
		}

		input_paths.push_back(dp);
	}

	cudaSafeVectorToDev(devF.dev_input, input_paths.data(), devF.nInput);
		

	return devF;
}


cuDigraph::dev_family_of_paths cuDigraph::FamilyOfPaths::load(family_size mOutput, family_size initial_allocation)
{
	assert(mOutput > 0);

	dev_family_of_paths devF;

	devF.resInput = std::max<family_size>({ (family_size) size()*mOutput, initial_allocation });
	devF.resOutput = devF.resInput;
	devF.nInput = (family_size)size();
	devF.mOutput = mOutput;

	cudaSafeMallocVector(devF.dev_input, devF.resInput);
	cudaSafeMallocVector(devF.dev_output, devF.resOutput);
	
	devF.nInput = (family_size)R.size();
	devF.clearOutput();

	cudaSafeVectorToDev(devF.dev_input, R.data(), devF.nInput);


	return devF;
}



namespace cuDigraph {

	

	FamilyOfPathsSTL augmentFamilyOfPaths(FamilyOfPathsSTL& F, dev_incidence_record& I, int blocks, int threads) {
		FamilyOfPathsSTL Fprime;

		VERBOSE(std::cout << "packing " << std::endl);

		/* load family onto device */
		auto devF = F.load(I.maxInDegree);

		VERBOSE(std::cout << "done." << std::endl);

		
		for (auto batch = 0; batch <= devF.nInput / (blocks*threads); ++batch) {
			family_size n0 = batch*(blocks*threads);

			VERBOSE(std::cout << "CUDA" << std::flush);
			
			/* call the kernel to augment the paths */
			augment_paths(blocks, threads, devF, I, 0, n0); // overflow is impossible since mOutput == maxInDegree

			cudaSafeCallNoExit(cudaDeviceSynchronize());
			cudaCheckErr("augment_paths(..)");
			VERBOSE(std::cout << " finished" << std::endl);
		}

		
		/* turn sparse results into compact array */

		compact_family_of_paths_output_to_input(devF);
		VERBOSE(std::cout << "thrust compacted." << std::endl);

		/* copy results back to host */

		std::vector<dev_walk> output;


		output.resize(devF.nInput);
		cudaSafeVectorToHost(output.data(), devF.dev_input, devF.nInput);

		VERBOSE(std::cout << "transferred to host." << std::endl);
		
		/* clean up device */
		devF.unload();
		
		/* convert to family of paths */
		/* This part seems to consume a lot of time, try to avoid it...*/

		for (auto p : output) {
			Fprime.R.insert(Walk(std::vector<vertex>(p.v, p.v + p.n)));
		}

		VERBOSE(std::cout << "unpacked." << std::endl);

		
		


		return Fprime;
	}

}