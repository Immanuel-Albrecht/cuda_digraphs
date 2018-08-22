# Digraph Combinatorics with CUDA

This repository contains CUDA code that may be fit for solving combinatorial problems on directed graphs using the massive parallelism of CUDA graphics cards. This is not a library and it is also not intended to be anytime soon. Instead I'm sharing useful code that I had to write in order to solve some problems I'm interested in; but feel free to give any advice or make improvements to it -- after all, that's the basic concept behind GPL, isn't it.

## Computing representations of directed paths in biorientations of complete graphs

I used the following code in order to test performance of the CUDA algorithms on a host with an AMD FX(tm)-6300 Six-Core Processor (3500 Mhz, 3 Cores, 6 Logical Processors) and 16GiB of RAM, and a (dedicated) NVIDIA GeForce GTX 1060 6GB graphics card for CUDA calculations.

```cpp
        int rtv, drv;
	cudaRuntimeGetVersion(&rtv);
	cudaDriverGetVersion(&drv);
	const size_t A007526[] = { 0, 1, 4, 15, 64, 325, 1956, 13699, 109600, 986409, 9864100, 108505111, 1302061344, 16926797485 }; // https://oeis.org/A007526
	for (auto N = 0; N <= 10; ++N) {
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
```

And here come the timing results (wall-time):
```txt
CUDA Runtime Version = 9000 Driver Version = 9020
K(0) paths:0 in 0 milliseconds. (= OEIS A007526[0])
K(1) paths:1 in 817 milliseconds. (= OEIS A007526[1])
K(2) paths:4 in 134 milliseconds. (= OEIS A007526[2])
K(3) paths:15 in 134 milliseconds. (= OEIS A007526[3])
K(4) paths:64 in 134 milliseconds. (= OEIS A007526[4])
K(5) paths:325 in 136 milliseconds. (= OEIS A007526[5])
K(6) paths:1956 in 137 milliseconds. (= OEIS A007526[6])
K(7) paths:13699 in 160 milliseconds. (= OEIS A007526[7])
K(8) paths:109600 in 327 milliseconds. (= OEIS A007526[8])
K(9) paths:986409 in 2273 milliseconds. (= OEIS A007526[9])
K(10) paths:9864100 in 24099 milliseconds. (= OEIS A007526[10])
```

The spike for K(1) is due to the late loading of the CUDA drivers; K(11) would roughly need 100GB of memory, but I did not implement any virtualization for host memory, so K(11) is out of reach. The roughly 24.1 seconds for generating a representation of every path in K(10) in host memory does not appear to be terribly slow.
