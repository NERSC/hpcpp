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
#include "sw-adept.hpp"

using namespace sw;

// ------------------------------------------------------------------------------------------------------------------------- //
//
// getSizes
//
[[nodiscard]] short* getSizes(scheduler auto &&sch, const std::vector<string_t> &database,
                              const std::vector<string_t> &queries)
{
    // N = size
    int N = database.size();

    // allocate array
    short *y = new short[N+1];

   // number of iterations
    int niters = ilog2(N);

    // get pointers
    auto db = database.data();
    auto q = queries.data();

    // write the minimum lengths of sequence from database and queries to y
    ex::sync_wait(ex::schedule(sch) | ex::bulk(N, [=](int k) { y[k] = std::min(db[k].length(), q[k].length()); }));

    // ---------------------------------------------------------------------------------------- //
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

    // ---------------------------------------------------------------------------------------- //
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

    // return y
    return y;
}

// ------------------------------------------------------------------------------------------------------------------------- //
//
// dna aligner kernel
//
// [[nodiscard]] dna_align();

// ------------------------------------------------------------------------------------------------------------------------- //
//
// aa aligner kernel
//

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

    if (scheduler == (sch_t)(-1))
        throw std::runtime_error("Run: `sw-adept-stdexec --help` to see the list of available schedulers");

    // ---------------------------------------------------------------------------------------- //
    //
    // parse input files and construct data structures
    //

    std::vector<string_t> database;
    std::vector<string_t> queries;

    // get max_query_length
    const auto max_query_len = (seq == seq_type_t::aa)? aa::MAX_QUERY_LEN : dna::MAX_QUERY_LEN;

    try {
        readFASTAs(dbFile, qFile, database, queries, max_query_len);
    }
    catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // ---------------------------------------------------------------------------------------- //
    //
    // get thread map
    //

    short *tmap = nullptr;

    // launch with appropriate stdexec scheduler
    switch (scheduler) {
        case sch_t::CPU:
            tmap = getSizes(exec::static_thread_pool(nthreads).get_scheduler(), database, queries);
            break;
#if defined(USE_GPU)
        case sch_t::GPU:
            tmap = getSizes(nvexec::stream_context().get_scheduler(), database, queries);
            break;
        case sch_t::MULTIGPU:
            tmap = getSizes(nvexec::multi_gpu_stream_context().get_scheduler(), database, queries);
            break;
#endif  // USE_GPU
        default:
            (void)(tmap);
    }


    // ---------------------------------------------------------------------------------------- //
    //
    // initialize scores, gaps and other properties
    //

    // scoring matrix
    std::vector<short> scores(aa::blosum62size);

    // assign blosum62 matrix to scores if amino acids
    if (seq == seq_type_t::aa)
        scores.assign(aa::blosum62, aa::blosum62 + aa::blosum62size);
    else
        scores.assign({MATCH, MISMATCH});

    // gaps matrix
    std::vector<short> gaps{GAP_OPEN, GAP_EXTEND};

    // ---------------------------------------------------------------------------------------- //
    //
    // launch the smith waterman kernel
    //

    // launch the smith waterman kernel

    // validate the output

    // return status
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------------- //
