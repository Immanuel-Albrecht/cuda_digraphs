// A simple program that computes the square root of a number
#include "pch.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>



#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "digraph.h"



int main(int argc, char *argv[])
{
	using namespace std;

	int rtv, drv;
	cudaRuntimeGetVersion(&rtv);
	cudaDriverGetVersion(&drv);

	cout << "CUDA Runtime Version = " << rtv << " Driver Version = " << drv << endl;

	/* Let's generate all paths in some complete graphs. */

	const size_t A007526[] = { 0, 1, 4, 15, 64, 325, 1956, 13699, 109600, 986409, 9864100, 108505111, 1302061344, 16926797485 }; // https://oeis.org/A007526

	for (auto N = 0; N <= 10; ++N) {
		/* for N==11, you probably need around 100GB to store all paths; 
		whereas N==10 needs <10GB to store.
		You could still calculate all the paths on the fly though, and throw them away after doing
		calculations.
		*/

		
		auto D2 = cuDigraph::Digraph();
		D2.addClique(0, N-1);

		auto t4 = std::chrono::high_resolution_clock::now();

		auto fastF3 = D2.allPaths();

		auto t5 = std::chrono::high_resolution_clock::now();
		cout << "K(" << (N) << ") paths:" << fastF3.size() << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << " milliseconds.";
		if (A007526[N] == fastF3.size()) {
			cout << " (= OEIS A007526[" << N << "])" << endl;
		}
		else {
			cout << "!!! ERROR !!!" << " #paths should be " << A007526[N] << " (= OEIS A007526[" << N << "])" << endl;
		}

	}
	
  return 0;
}