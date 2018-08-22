#pragma once

#include "pch.h"



namespace cuDigraph {

	/* SETUP CONSTANTS */
	static const int max_path_length = 128; //max. number of vertices traversed by a path


	/* FORWARD DECLARATIONS */

	class Walk;
	class FamilyOfPathsSTL;
	class FamilyOfPaths;
	class Digraph;


	/* CODE PART */


	typedef int vertex; // vertex data type
	typedef std::pair< vertex, vertex >   arc; // arc data type **tail-->head**; a.first == head, a.second == tail;

	typedef int index; // indexing integer type for incident lists
	typedef index size; // size integer type for incident lists

	typedef uint64_t elt_vertex_set; // vertex set element
	static const int per_vertex_set_elt = sizeof(elt_vertex_set) * 8; //number of bits in each vertex set elt.
	typedef elt_vertex_set	vertex_set[4]; // small set of vertices
	static const int max_vertex_set_vertex_id = sizeof(vertex_set) * 8; //bits in each vertex set

	/* test whether vertex v is in vertex set S */
	__host__ __device__ inline bool  is_in_vertex_set(const vertex_set& S, vertex v) {
		int idx = (v / per_vertex_set_elt);
		int bit = v - (idx*per_vertex_set_elt);
		return (S[idx] & (((elt_vertex_set)1) << bit)) != 0;
	}
	__host__ __device__ inline void  set_in_vertex_set(vertex_set& S, vertex v) {
		int idx = (v / per_vertex_set_elt);
		int bit = v - (idx*per_vertex_set_elt);
		S[idx] |= (((elt_vertex_set)1) << bit);
	}
	__host__ __device__ inline void  clear_in_vertex_set(vertex_set& S, vertex v) {
		int idx = (v / per_vertex_set_elt);
		int bit = v - (idx*per_vertex_set_elt);
		S[idx] &= ~(((elt_vertex_set)1) << bit);
	}
	__host__ __device__ inline void  zero_out_vertex_set(vertex_set& S) {
		for (auto i = 0; i < sizeof(vertex_set) / sizeof(elt_vertex_set); ++i) {
			S[i] = 0;
		}
	}
	__host__ __device__ inline void  copy_vertex_set(const vertex_set& src, vertex_set& dst) {
		for (auto i = 0; i < sizeof(vertex_set) / sizeof(elt_vertex_set); ++i) {
			dst[i] = src[i];
		}
	}
	__host__ __device__ inline bool  has_empty_intersection_vertex_set(const vertex_set& S, const vertex_set& T) {
		for (auto i = 0; i < sizeof(vertex_set) / sizeof(elt_vertex_set); ++i) {
			if ((S[i] & T[i]) != 0)
				return false;
		}
		return true;
	}
	__host__ __device__ inline bool  extend_vertex_set(vertex_set& S, const vertex_set& T) {
		for (auto i = 0; i < sizeof(vertex_set) / sizeof(elt_vertex_set); ++i) {
			S[i] |= T[i];
		}
		return true;
	}
	__host__ inline int cardinality_of_vertex_set(const vertex_set& S) {
		int card = 0;
		for (auto i = 0; i < sizeof(vertex_set) / sizeof(elt_vertex_set); ++i) {
			card += (int)popcnt64(S[i]); /* popcnt64 does not run on the CUDA device! */
		}
		return card;
	}

	typedef struct incidence_record_ {
		std::vector<index> start; // start index of list for vertex [v].
		std::vector<size>  length; // length of list for vertex [v].
		std::vector<vertex> incidence; // collated in-neighbor incidence data; start  refers to **heads** of arcs;

		bool isValid() {
			if (start.size() != length.size())
				return false;
			for (auto i = 0; i < start.size(); ++i) {
				if (start[i] + length[i] > incidence.size())
					return false; /* memory corruption!! */
			}
			return true;
		}
	}
	incidence_record; // stores the arcs of a digraph

	typedef struct dev_incidence_record_ {
		vertex nV; // number of vertices, size of index and length lists
		index *start; // start indexes
		size  *length; // length of incidence list
		vertex *incidence; // collated in-neighbor incidence data

		size maxInDegree; // maximum in-degree of the incidence record.

		void unload(); // frees the device memory from this structure
	} dev_incidence_record;

	/*
	 *   DiGraph abstraction class
	 *
	 */
	class Digraph {
	public:

		std::set<vertex> V; //vertices
		std::set<arc> A; // arcs, first element is **head**, second element is **tail**; arc goes **tail --> head**.

		Digraph() {}; // constructor

		/* isValid
		 *
		 * checks, whether V contains all vertices which have arcs
		 * and whether all elements of V are non-negative.
		 *
		 * returns true; if the object has valid structure; false otherwise.
		 */
		bool isValid() const {
			for (auto a : A) {
				if ((V.find(a.first) == V.end()) || (V.find(a.second) == V.end())) {
					// Found an arc with a 'dangling' vertex
					return false;
				}
			}
			for (auto v : V)
			{
				if (v < 0)
				{
					// Found a negative vertex!
					return false;
				}
			}
			return true;
		}

		/* fixVfromA
		 *
		 *  adds all vertices that occur is some arc from A
		 *  to the set V.
		 *
		 */
		void fixVfromA() {
			for (auto a : A) {
				V.insert(a.first);
				V.insert(a.second);
			}
		}

		/* addArc
		 *
		 * adds an arc from tail -> head.
		 */
		void addArc(vertex tail, vertex head) {
			V.insert(head);
			V.insert(tail);

			A.insert(arc(head, tail));
		}

		/* addClique
		 *
		 * adds a contiguous clique (v,v+1,v+2,...,u) where v == start and u == end.
		 *
		 */
		void addClique(vertex start, vertex end) {
			if (start == end) {
				V.insert(start);
				return;
			}
			for (vertex v = start; v <= end; ++v) {
				for (vertex u = v + 1; u <= end; ++u) {
					addArc(u, v);
					addArc(v, u);
				}
			}
		}
		/* addClique
		*
		* adds a clique given by the input parameter
		*
		*/
		void addClique(std::vector<vertex> clique) {
			if (clique.size() == 1) {
				V.insert(clique[0]);
				return;
			}
			for (auto i = 0; i < clique.size(); ++i)
			{
				for (auto j = i + 1; j < clique.size(); ++j) {
					addArc(clique[i], clique[j]);
					addArc(clique[j], clique[i]);
				}

			}
		}



		/* maxInDegree
		 *
		 * returns the maximum in-degree of the vertices of the digraph.
		 *
		 */
		size maxInDegree() const {
			std::map<vertex, size> count;

			size max = 0;

			for (auto a : A) {
				count[a.first] += 1;
				if (count[a.first] > max)
					max = count[a.first];
			}

			return max;
		}

		/*
		 * compactify
		 *
		 *  compacts the digraph by removing isolated vertices and relabelling
		 *  the remaining vertices to 0..n
		 *
		 */
		void compactify() {

			// remove isolated vertices
			V.clear();
			fixVfromA();

			// rename incident vertices
			vertex id = 0;

			std::map<vertex, vertex> relabelling;
			std::set<vertex> relabelledV;

			for (auto v : V) {
				relabelledV.insert(id);
				relabelling[v] = id++;
			}

			std::set<arc> relabelledA;


			for (auto a : A) {
				arc a2(relabelling[a.first], relabelling[a.second]);
				relabelledA.insert(a);
			}

			//overwrite old structure
			A = relabelledA;
			V = relabelledV;
		}

		/* encode
		 *
		 * creates an incidence record for the given digraph,
		 * which in turn may be fed to the CUDA part.
		 *
		 * The object must be valid!
		 * (you should compactify if possible)
		 */
		incidence_record encode() const {

			assert(this->isValid());
			incidence_record IR;

			if (V.empty())
				return IR;

			vertex Vmax = *V.rbegin(); //maximal vertex index



			IR.start.resize(Vmax + 1);
			IR.length.resize(Vmax + 1);
			IR.incidence.reserve(A.size());

			vertex previous_head = 0;
			for (auto a : A) {
				if (a.first != previous_head) {
					IR.start[a.first] = (size)IR.incidence.size();
					previous_head = a.first;
				}
				IR.incidence.push_back(a.second);
				IR.length[a.first] ++;
			}

			assert(IR.incidence.size() == std::accumulate(IR.length.begin(), IR.length.end(), 0));

			return IR;
		}

		/* load
		 *
		 * loads the digraph incidence structure onto the CUDA device.
		 * The returned record must be unloaded by the caller.
		 *
		 */
		dev_incidence_record load() const;

		/* sinks
		 *
		 * returns the set of sink vertices in the digraph
		 */
		std::set<vertex> sinks() const {
			std::set<vertex> sinks(V.begin(), V.end());

			for (auto a : A) {
				sinks.erase(a.second); /* arc a: a.second --> a.first*/
			}

			return sinks;
		}

		/* allPathsTo [slow version]
		 *
		 * returns all paths that end in a given set of vertices;
		 *         given as begin and end iterators
		 */
		template<typename Iter> FamilyOfPathsSTL stl_allPathsTo(Iter begin, Iter end, int max_len = -1, int blocks = 1024, int threads = 1024) const;

		FamilyOfPathsSTL stl_allPaths(int max_len = -1, int blocks = 1024, int threads = 1024) const;

		template<typename Iter> FamilyOfPaths allPathsTo(Iter begin, Iter end, int max_len = -1, int initial_allocation_size = 40320 /* factorial(9)*/, int maximal_device_size = -1, int blocks = 1024, int threads = 1024) const;

		FamilyOfPaths allPaths(int max_len = -1, int initial_allocation_size = 40320 /* factorial(9)*/,
			int maximal_device_size = -1, int blocks = 1024, int threads = 1024) const;


	};

	inline std::ostream& operator<< (std::ostream& o, const Digraph& D) {

		o << "Digraph with " << D.V.size() << " vertices and " << D.A.size() << " arcs." << std::endl;

		for (auto v : D.V) {
			o << v << "->: ";

			bool first = true;
			for (auto a : D.A) {
				if (a.second == v) {
					if (!first)
						o << ", ";

					o << a.first;

					first = false;
				}
			}
			o << std::endl;
		}

		return o;
	}

	typedef int path_size;

	/* structure that stores a path (for use on the CUDA device) [technically, it stores walks though] */
	typedef struct dev_walk_ {
		path_size n; // number of vertices visited by the walk; trivial paths have n=1, n=0 is used to indicate that the struct is not used
		vertex_set support; // vertices visited by the walk, as set (for quick disjoint vertex set testing)
		vertex v[max_path_length]; // vertices visited by the walk, in the correct order.

		__host__ __device__ dev_walk_&  operator=(const dev_walk_& that)
		{
			/* quite surprisingly, the default operator of this plain old data struct
			   is not behaving well with thrust::copy_if and corrupts memory. This version does not.
			   */
			n = that.n;
			copy_vertex_set(that.support, support);
			for (auto i = 0; i < n; ++i)
			{
				v[i] = that.v[i];
			}

			return *this;
		}

		/* tests whether this dev_path struct is indeed a path and not merely a walk*/
		__host__ bool isPath() const {
			return n == cardinality_of_vertex_set(support);
		}
	} dev_walk;

	inline std::ostream& operator<< (std::ostream& o, const dev_walk& w) {
		if (w.n == 0)
			o << "(eps)";
		bool first = true;

		for (auto i = 0; i < w.n; ++i) {

			auto x = w.v[i];

			if (!first)
				o << "-";

			o << x;

			first = false;
		}
		return o;
	}

	/*
	 *  Walk   abstraction class for paths (and walks) in digraphs (no multiple edges between vertices)
	 *
	 *
	 *
	 */
	class Walk {
	public:
		std::vector<vertex> v; // stores the vertices of the path in order of traversal
		std::set<vertex> support; // stores the support of the path, i.e. all vertices traversed

		Walk() {}; //constructor

		Walk(std::vector<vertex> v0) : v(v0.begin(), v0.end()) {
			fixSupport();
		}

		template<typename Iter> Walk(Iter begin, Iter end) : v(begin, end) {
			fixSupport();
		}

		/* check whether support is correct for v*/
		bool isValid() const {
			std::set<vertex> sup2(v.begin(), v.end());

			return sup2 == support;
		}

		/* test whether this is the empty walk, which technically is not a walk.*/
		bool isEmpty() const {
			return v.size() == 0;
		}

		/* test whether this is the trivial walk for some vertex v[0]. */
		bool isTrivial() const {
			return v.size() == 1;
		}

		/* update support member variable */
		void fixSupport() {
			support = std::set<vertex>(v.begin(), v.end());
		}

		/* test whether the walk is a path */
		bool isPath() const {
			return v.size() == support.size();
		}


	};

	/* Comparison operations for Walks */

	inline bool operator==(const Walk& lhs, const Walk& rhs) { return lhs.v == rhs.v; }
	inline bool operator!=(const Walk& lhs, const Walk& rhs) { return !(lhs == rhs); }
	inline bool operator< (const Walk& lhs, const Walk& rhs)
	{
		size_t i = 0;
		do {
			if (lhs.v.size() <= i)     return  (rhs.v.size() > i);
			if (rhs.v.size() <= i)     return  false;
			if (lhs.v[i] < rhs.v[i])   return  true;
			if (lhs.v[i] > rhs.v[i])   return  false;
			++i;
		} while (1);
	}
	inline bool operator> (const Walk& lhs, const Walk& rhs) { return rhs < lhs; }
	inline bool operator<=(const Walk& lhs, const Walk& rhs) { return !(lhs > rhs); }
	inline bool operator>=(const Walk& lhs, const Walk& rhs) { return !(lhs < rhs); }
	inline std::ostream& operator<< (std::ostream& o, const Walk& w) {
		if (w.v.empty())
			o << "(eps)";
		bool first = true;

		for (auto x : w.v) {
			if (!first)
				o << "-";

			o << x;

			first = false;
		}
		return o;
	}


	typedef int family_size;

	/* structure for storing paths on the CUDA device */
	typedef struct dev_family_of_paths_ {
		family_size nInput;   // number of input path slots
		family_size resInput; // number of input path slots allocated
		dev_walk *dev_input;  // input paths on the device


		family_size mOutput;   // size of each output path bucket
		family_size resOutput; // number of output path slots allocated
		dev_walk *dev_output;  // output path targets on the device


		/* tests whether the memory layout is valid; if false is returned, then there is a good chance of memory corruption*/
		bool isValid() {
			if (nInput > resInput)
				return false;
			return nInput * mOutput <= resOutput;
		}

		/* make the output size at least nInput*mOutput; this routine does not copy old output data*/
		void enlargeOutput();
		/* make the input size at least nInput*mOutput;
		this routine copies the first nInput paths to the new location, if necessary */
		void enlargeInput();

		/* complete enlargement cycle; destroys any output data, but conserves input data.
		-- prepares for another run on the device using the results of the last run. */
		void enlargeAndPrepare() {
			enlargeOutput();
			clearOutput();
			enlargeInput();
		}

		/* test whether an additional enlargement would break through the given limit */
		bool enlargeBreaksLimit(family_size limit) {
			return (nInput * mOutput > limit);
		}

		/* enlarges the structure to the given limit; does not preserve any stored data */
		void enlargeToLimit(family_size limit);

		void unload();
		void clearOutput();
		void clearInput();

	} dev_family_of_paths;

	/*
	* FamilyOfPaths
	*
	* class that represents a family of paths in a digraph,
	* where each path is considered to be a sequence of vertices;
	*
	* this class avoids STL for performance reasons
	*
	* technically, you could store walks in here, but that is not what we are interested in
	*
	*/
	class FamilyOfPaths {
	public:
		std::vector<dev_walk> R; // stores the family as dev_path's; this means faster i/o action

		FamilyOfPaths() {};

		/* adds all trivial paths for a given set of vertices to the family of paths */
		void addTrivialPaths(const std::set<vertex>& V) {
			addTrivialPaths(V.begin(), V.end());
		}
		/* adds all trivial paths for a given set of vertices to the family of paths */
		template<typename Iter> void addTrivialPaths(Iter begin, Iter end) {
			dev_walk w;
			w.n = 1;
			for (; begin != end; ++begin) {
				w.v[0] = *begin;
				zero_out_vertex_set(w.support);
				set_in_vertex_set(w.support, w.v[0]);

				R.push_back(w);
			}
		}

		/* size of this family */
		auto size() const {
			return R.size();
		}

		/* test whether this family is a routing */
		bool isFamilyOfPaths() const {
			for (auto w : R) {
				if ((!w.isPath()) || (w.n == 0))
					return false;
			}
			return true;
		}

		/* test whether all walks are pair-wise vertex disjoint */
		bool isPairwiseVertexDisjoint() const {
			vertex_set X; /* vertices that are in the support of some path considered before */
			zero_out_vertex_set(X);

			for (auto w : R) {
				if (!has_empty_intersection_vertex_set(X, w.support))
					return false;

				/* add support of w to X */
				extend_vertex_set(X, w.support);
			}

			return true;
		}

		/* mOutput should be >= maximum in-degree of the graphs involved;
		   initial_allocation may be used to over-allocate (reuse scenario)
		   */
		dev_family_of_paths load(family_size mOutput, family_size initial_allocation = -1);


	};

	inline std::ostream& operator<< (std::ostream& o, const FamilyOfPaths& F) {
		o << "{";
		bool first = true;

		for (auto w : F.R) {
			if (!first)
				o << ", ";

			o << w;

			first = false;
		}
		o << "}";
		return o;
	}



	/*
	 * FamilyOfPathsSTL
	 *
	 * class that represents a family of paths in a digraph,
	 * where each path is considered to be a sequence of vertices;
	 *
	 * technically, you could store walks in here, but that is not what we are interested in
	 *
	 * this class has the maximum convenience of the STL, but gets slow sooner
	 */
	class FamilyOfPathsSTL {
	public:
		std::set< Walk > R; // stores a family of paths, a (pre)-routing

		FamilyOfPathsSTL() {}; //constructor

		/* adds all trivial paths for a given set of vertices to the family of paths */
		void addTrivialPaths(const std::set<vertex>& V) {
			addTrivialPaths(V.begin(), V.end());
		}
		/* adds all trivial paths for a given set of vertices to the family of paths */
		template<typename Iter> void addTrivialPaths(Iter begin, Iter end) {
			for (; begin != end; ++begin) {
				R.insert(Walk({ *begin }));
			}
		}

		/* size of this family */
		auto size() const {
			return R.size();
		}

		/* test whether all walks are pair-wise vertex disjoint */
		bool isPairwiseVertexDisjoint() const {
			std::set<vertex> X; /* vertices that are in the support of some path considered before */

			for (auto w : R) {
				/* well, there should be std::empty_intersection as a spinoff of std::set_intersection, yet there isn't.

					-> CODE DIVE
				*/

				auto i = X.begin();
				auto j = w.support.begin();
				auto iN = X.end();
				auto jN = w.support.end();

				while (i != iN && j != jN)
				{
					if (*i == *j) {
						return false; /* X and w.support have non-empty intersection here.*/
					}
					else if (*i < *j)
						++i;
					else
						++j;
				}

				/* add support of w to X */
				X.insert(w.support.begin(), w.support.end());
			}

			return true;
		}

		/* test whether this family is a routing */
		bool isFamilyOfPaths() const {
			for (auto w : R) {
				if ((!w.isPath()) || (w.isEmpty()))
					return false;
			}
			return true;
		}

		/*
		 * load
		 *
		 * loads the current family of paths onto the CUDA device
		 *
		 * mOutput       __ size of each output batch (using something above maxInDegree is usually a safe choice)
		 * allocOutput   __ number of output paths to allocate
		 * allocInput    __ number of input paths to allocate
		 * symmetric     __ if true, then the allocated sizes for input and output are matched
		 *
		 * the returned dev_family_of_paths must be unloaded by the caller.
		 */
		dev_family_of_paths load(family_size mOutput, family_size allocOutput = -1, family_size allocInput = -1, bool symmetric = true);
	};

	inline std::ostream& operator<< (std::ostream& o, const FamilyOfPathsSTL& F) {
		o << "{";
		bool first = true;

		for (auto w : F.R) {
			if (!first)
				o << ", ";

			o << w;

			first = false;
		}
		o << "}";
		return o;
	}

	/* augmentFamilyOfPaths

	   returns a family of paths where each path
			   consists of a path from F plus a new initial vertex according to I;
			   runs on a CUDA device.
	*/
	FamilyOfPathsSTL augmentFamilyOfPaths(FamilyOfPathsSTL & F, dev_incidence_record & I, int blocks = 1024, int threads = 512);



	/* copies the output to the input of the devF; asserts that resInput >= nInput*mOutput,
		uses thrust, caller has to synchronize CUDA before results are available!
	*/
	void compact_family_of_paths_output_to_input(dev_family_of_paths& devF);

	/* initiates the kernel to be run; caller has to synchronize CUDA before results are available! */
	void augment_paths(int blocks, int threads, dev_family_of_paths& F, dev_incidence_record& I, bool * batchOverflow, family_size n0);

}

template<typename Iter>
inline cuDigraph::FamilyOfPathsSTL cuDigraph::Digraph::stl_allPathsTo(Iter begin, Iter end, int max_len, int blocks, int threads) const
{
	FamilyOfPathsSTL all, current;

	current.addTrivialPaths(begin, end);

	auto I = load();

	while (current.R.size() > 0) {
		all.R.insert(current.R.begin(), current.R.end());

		if (max_len == 0)
			break;


		current = augmentFamilyOfPaths(current, I, blocks, threads);
		max_len--; /* this will never underrun before we are out of memory */
	}

	I.unload();

	return all;
}

template<typename Iter>
inline cuDigraph::FamilyOfPaths cuDigraph::Digraph::allPathsTo(Iter begin, Iter end, int max_len, int initial_allocation_size, int maximal_device_size , int blocks, int threads) const
{
	FamilyOfPaths F;
	auto output_index = F.size();

	F.addTrivialPaths(begin, end);

	if (F.size() == 0)
	{
		return F;
	}




	VERBOSE(std::cout << "loading." << std::endl);

	auto I = load();

	if (I.maxInDegree == 0)
		I.maxInDegree = 1; /* prevent division by zero. */

	if (maximal_device_size < 0) {
		/* calculate the maximal device size from current available memory */
		size_t dev_free, dev_max;
		cudaSafeCall(cudaMemGetInfo(&dev_free, &dev_max));

		/* we choose an 50% of free memory strategy here;
		   we need some memory for thrust::copy_if*/

		maximal_device_size =(family_size) ((((5 * dev_free) / 10)) / (2 * sizeof(dev_walk)));

		VERBOSE(std::cout << "Device may hold two " 
			<<	maximal_device_size
			<< " path arrays." << std::endl);
	}

	if (maximal_device_size < I.maxInDegree) {
		maximal_device_size = I.maxInDegree; /* the bare minimum possible */
	}

	if (maximal_device_size < initial_allocation_size) {
		initial_allocation_size = maximal_device_size;
	}

	auto maximal_input_chunk_size = maximal_device_size / I.maxInDegree;

	auto devF = F.load(I.maxInDegree, initial_allocation_size);

	assert(maximal_input_chunk_size * devF.mOutput <= maximal_device_size);

	while (output_index < F.size()) {
			if (max_len == 0)
				break;
		VERBOSE(std::cout << "input batch size = " << (F.size() - output_index) << std::endl);

		/* if necessary, increase the amount of memory reserved on the CUDA device */
		
		if ((output_index - F.size()) > maximal_input_chunk_size) {
			VERBOSE(std::cout << "device memory limit hit." << std::endl);

			/* enlarge to limit */
			devF.enlargeToLimit(maximal_device_size);

			auto start_of_current_chunk = output_index;
			auto end_of_current_batch = F.size();

			

			auto chunk_count = 0;
			auto nbr_of_chunks = (end_of_current_batch - start_of_current_chunk + (maximal_input_chunk_size - 1)) / maximal_input_chunk_size;

			auto this_chunk_size = maximal_input_chunk_size;
			
			while (start_of_current_chunk < end_of_current_batch) {

				/* calculate the size of the current chunk */
				if (end_of_current_batch - start_of_current_chunk < maximal_input_chunk_size)
					this_chunk_size = (family_size) ( end_of_current_batch - start_of_current_chunk );
				

				/* we have to prepare the input data (always!) */
				VERBOSE(std::cout << "Moving input to device." << std::endl);
				cudaSafeVectorToDev(devF.dev_input, F.R.data() + start_of_current_chunk, this_chunk_size);
				

				/* update the input size of the*/
				devF.nInput = this_chunk_size;

				/* clear the output data structure */
				devF.clearOutput();

				/* update the start of the current chunk */
				start_of_current_chunk += this_chunk_size;

				/* run the next chunk */

				++chunk_count;
				VERBOSE(std::cout << "chunk " << chunk_count << "/" << nbr_of_chunks << " [" << this_chunk_size << " paths]" << "." << std::endl);
								


				
				/* augment paths in devF on CUDA device */
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

				/* copy results to host */
				auto write_index = F.size();

				F.R.resize(write_index + devF.nInput);
				auto output_target = F.R.data() + write_index;

				cudaSafeVectorToHost(output_target, devF.dev_input, devF.nInput);
				VERBOSE(std::cout << "transferred to host." << std::endl);
			}

			assert(start_of_current_chunk == end_of_current_batch);

			/* set the output index such that it adresses the first element of the next batch. */
			output_index = start_of_current_chunk; 

			VERBOSE(std::cout << "output batch size = " << (F.size() - start_of_current_chunk) << std::endl);

			continue; /* continue the main loop */
		}

		output_index = F.size();

		/* apparently, everything still fits into CUDA device memory */
				
		VERBOSE(std::cout << "enlarge devF." << std::endl);
		devF.enlargeAndPrepare();

		/* augment paths in devF on CUDA device */
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

		/* copy results to host */

		F.R.resize(output_index + devF.nInput);
		auto output_target = F.R.data() + output_index;

		cudaSafeVectorToHost(output_target, devF.dev_input, devF.nInput);
		VERBOSE(std::cout << "transferred to host." << std::endl);

		VERBOSE(std::cout << "output batch size = " << (devF.nInput) << std::endl);

		max_len--;
	}


	VERBOSE(std::cout << "unloading." << std::endl);

	devF.unload();
	I.unload();

	return F;
}
