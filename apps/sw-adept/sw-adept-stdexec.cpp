/*
 * MIT License
 *
 * Copyright (c) 2023 The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of any
 * required approvals from the U.S. Dept. of Energy).All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 * commons for the smith waterman codes
 */
#define SW_STDEXEC
#include "adept.hpp"

using namespace sw;
using algn_t = sw::alignments;

// ------------------------------------------------------------------------------------------------------------------------- //
//
// binary Search kernel
//
int binarySearch(int *thread_map, int k, int N)
{
    int cell = 0;
    int left = 0;
    int right = N;

    for (;;)
    {
        int mid = left + (right - left) / 2;

        if (thread_map[mid] <= k)
        {
            if (thread_map[mid+1] > k)
            {
                cell = mid;
                break;
            }
            else
                left = mid+1;
        }
        else
            right = mid;
    }

    return cell;
}

// ------------------------------------------------------------------------------------------------------------------------- //
//
// prefixSum kernel
//
template <typename T>
void prefixSum(scheduler auto &&sch, T *y, const int N)
{
    // number of iterations
    int niters = ilog2(N);

    // GE Blelloch (1990) algorithm
    // upsweep
    for (int d = 0; d < niters; d++) {
        int bsize = N / (1 << d + 1);

        ex::sender auto uSweep = schedule(sch) | ex::bulk(bsize, [=](int k) {
                                     // stride1 = 2^(d+1)
                                     int st1 = 1 << d + 1;
                                     // stride2 = 2^d
                                     int st2 = 1 << d;
                                     // only the threads at indices (k+1) * 2^(d+1) -1 will compute
                                     int myIdx = (k + 1) * st1 - 1;

                                     // update y[myIdx]
                                     y[myIdx] += y[myIdx - st2];
                                 });
        // wait for upsweep
        ex::sync_wait(uSweep);
    }

    // write sum to y[N] and reset vars
    ex::sync_wait(schedule(sch) | ex::then([=]() {
                      y[N] = y[N - 1];
                      y[N - 1] = 0;
                  }));

    // downsweep
    for (int d = niters - 1; d >= 0; d--) {
        int bsize = N / (1 << d + 1);
        ex::sender auto dSweep = schedule(sch) | ex::bulk(bsize, [=](int k) {
                                     // stride1 = 2^(d+1)
                                     int st1 = 1 << d + 1;
                                     // stride2 = 2^d
                                     int st2 = 1 << d;
                                     // only the threads at indices (k+1) * 2^(d+1) -1 will compute
                                     int myIdx = (k + 1) * st1 - 1;

                                     // update y[myIdx] and y[myIdx-stride2]
                                     auto tmp = y[myIdx];
                                     y[myIdx] += y[myIdx - st2];
                                     y[myIdx - st2] = tmp;
                                 });

        // wait for downsweep
        ex::sync_wait(dSweep);
    }
}

// ------------------------------------------------------------------------------------------------------------------------- //
//
// buildThreadMap
//
void buildThreadMap(scheduler auto sch, std::vector<string_t> &database,
                                std::vector<string_t> &queries, int *y, const int N)
{
    // allocate vector
    if (!y)
        y = new int[N+1];

    // get pointers
    auto db = database.data();
    auto q = queries.data();

    // write the minimum lengths of sequence from database and queries to y
    ex::sync_wait(ex::schedule(sch) | ex::bulk(N,
        [=](int k) { y[k] = std::min(db[k].length(), q[k].length()); }));

    // compute prefixSum
    prefixSum(sch, y, N);

}

// ------------------------------------------------------------------------------------------------------------------------- //
//
// aligner kernel
//
void aligner(scheduler auto &&sch, std::vector<string_t> &database, std::vector<string_t> &queries, adept &driver)
{
    // database size
    const int N = database.size();

    // ---------------------------------------------------------------------------------------- //
    //
    // setup matrices
    //
    // scoring matrix
    std::vector<short> scores_vec(aa::blosum62size);
    (driver.getSeqType() == seq_type_t::aa)? scores_vec.assign(aa::blosum62, aa::blosum62 + aa::blosum62size) : scores_vec.assign({MATCH, MISMATCH});

    // gaps matrix
    std::vector<short> gaps_vec{GAP_OPEN, GAP_EXTEND};

    // encoding matrix
    std::vector<short> encode_vec(sw::encoding, sw::encoding + encode_mat_size);

    // get pointers to heap mem
    auto scores = scores_vec.data();
    auto gaps = gaps_vec.data();
    auto encode = encode_vec.data();

    // build the thread map
    int *thread_map = nullptr;
    buildThreadMap(sch, database, queries, thread_map, N);

    // ---------------------------------------------------------------------------------------- //
    //
    // dna alignment kernel
    //
    auto dna_kernel = [=](int k) {

        // get alignment number
        auto bin = binarySearch(thread_map, k, thread_map[N]);

    };

    // ---------------------------------------------------------------------------------------- //
    //
    // aa alignment kernel
    //
    auto aa_kernel = [=](int k){ auto bin = binarySearch(thread_map, k, thread_map[N]); };

    // ---------------------------------------------------------------------------------------- //
    //
    // launch kernel
    //
    // number of threads -> last element of thread_map
    int nthreads = thread_map[N];

    // launch the appropriate kernel
    (driver.getSeqType() == seq_type_t::aa) ? ex::sync_wait(ex::schedule(sch) | ex::bulk(nthreads, aa_kernel)) :
                                              ex::sync_wait(ex::schedule(sch) | ex::bulk(nthreads, dna_kernel));

}

// ------------------------------------------------------------------------------------------------------------------------- //
//
// main simulation
//
int main(int argc, char* argv[]) {

    // parse input parameters
    const sw_params_t args = argparse::parse<sw_params_t>(argc, argv);

    auto dbFile = args.db;
    auto qFile = args.data;
    auto sched = args.sch;
    const auto validFile = args.validate.value_or("");
    const auto seq = getSeqtype(args.seq);
    const auto cigar = args.cigar;
    const auto nthreads = args.nthreads;

    sch_t scheduler = get_sch_enum(sched);

    // ---------------------------------------------------------------------------------------- //
    //
    // initialize scores, gaps and other properties
    //
    auto driver = sw::adept(seq, cigar);

    // ---------------------------------------------------------------------------------------- //
    //
    // parse input files and construct data structures
    //
    // database matrices
    std::vector<string_t> database;
    std::vector<string_t> queries;

    try {
        driver.readFASTAs(dbFile, qFile, database, queries);
    }
    catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // ---------------------------------------------------------------------------------------- //
    //
    // launch the smith waterman kernel
    //
    switch (scheduler) {
        case sch_t::CPU:
            aligner(exec::static_thread_pool(nthreads).get_scheduler(), database, queries, driver);
            break;
    #if defined(USE_GPU)
        case sch_t::GPU:
            aligner(nvexec::stream_context().get_scheduler(), database, queries, driver);
            break;
        case sch_t::MULTIGPU:
            aligner(nvexec::multi_gpu_stream_context().get_scheduler(), database, queries, driver);
            break;
    #endif  // USE_GPU
        default:
            (void)(scheduler);
    }

    // ---------------------------------------------------------------------------------------- //
    //
    // validate the output
    //
    if (args.validate.has_value())
    {
        if (!driver.validate(validFile))
        {
            fmt::print("STATUS: Validation failed.\n");
        }
    }


    // return status
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------------- //
