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
// inclusive scan kernel
//
template <typename T>
void inclusiveScan(scheduler auto &&sch, T *y, const int N)
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
// build thread map and setup prefixes
//
int *setupDatabaseAndQueries(scheduler auto sch, std::vector<string_t> &database,
                                std::vector<string_t> &queries, adept &driver)
{
    const int N = database.size();

    // database and query lengths
    auto db_l = driver.len_db;
    auto q_l = driver.len_queries;

    // allocate memory for thread map
    int *tmap = new int[N+1];

    // get pointers
    auto db = database.data();
    auto q = queries.data();

    // write the minimum database and queries sequence lengths to y
    ex::sync_wait(ex::schedule(sch) | ex::bulk(N,
        [=](int k) { tmap[k] = std::min(db_l[k], q_l[k]); }));

    // inclusive scan on threadmap, db_l and q_l
    inclusiveScan(sch, tmap, N);
    inclusiveScan(sch, db_l, N);
    inclusiveScan(sch, q_l, N);

    return tmap;
}

// ------------------------------------------------------------------------------------------------------------------------- //
//
// aligner kernel
//
void aligner(scheduler auto &&sch, std::vector<string_t> &database, std::vector<string_t> &queries, adept &driver)
{
    // database size
    const int N = database.size();

    // reverse search
    bool reverse = false;

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

    // ---------------------------------------------------------------------------------------- //
    //
    // build the thread map and setup dynamic memories
    //
    int *thread_map =  setupDatabaseAndQueries(sch, database, queries, driver);

    // ---------------------------------------------------------------------------------------- //
    //
    // get pointers to heap mem
    //
    auto scores = scores_vec.data();
    auto gaps = gaps_vec.data();
    auto encode = encode_vec.data();

    // extract pointers from driver
    auto seqDB = driver.database;
    auto seqQ = driver.queries;

    auto lenDB = driver.len_db;
    auto lenQ = driver.len_queries;

    // ---------------------------------------------------------------------------------------- //
    //
    // dna alignment kernel
    //
    auto dna_kernel = [=](int k) {

        // get alignment number via binary search
        auto bin = binarySearch(thread_map, k, thread_map[N]);

        //
        // setup basic pointers
        //

        // get the current database and query sequences
        char *seqA = seqDB + lenDB[bin];
        int lenA = lenDB[bin+1] - lenDB[bin];
        char *seqB = seqQ + lenQ[bin];
        int lenB = lenQ[bin+1] - lenQ[bin];

        //
        // setup thread offset, longer_seq, and myColumnChar
        //

        // thread offset
        int thread_offset = k;
        thread_offset -= (lenA < lenB)? lenDB[bin] : lenQ[bin];

        // longer sequence
        char* longer_seq = (lenA < lenB)? seqB : seqA;

        // max and min lengths
        unsigned maxSize = max(lenA, lenB);
        unsigned minSize = min(lenA, lenB);

        //
        // move my char to register (local variable)
        //
        char myColumnChar;

        if(lenA < lenB)
            myColumnChar = (reverse)? seqA[(lenA - 1) - thread_offset] : seqA[thread_offset];
        else
            myColumnChar = (reverse)? seqB[(lenB - 1) - thread_offset] : seqB[thread_offset];

        /*
        int   i            = 1;
        short thread_max   = 0; // to maintain the thread max score
        short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
        short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string

        //initializing registers for storing diagonal values for three recent most diagonals (separate tables for
        //H, E and F)
        short _curr_H = 0, _curr_F = 0, _curr_E = 0;
        short _prev_H = 0, _prev_F = 0, _prev_E = 0;
        short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
        short _temp_Val = 0;
        */

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

    // launch the appropriate kernel (forward mode)
    (driver.getSeqType() == seq_type_t::aa) ? ex::sync_wait(ex::schedule(sch) |ex::bulk(nthreads, aa_kernel)) :
                                              ex::sync_wait(ex::schedule(sch) |  ex::bulk(nthreads, dna_kernel));

    reverse = true;
    // launch the appropriate kernel (reverse mode)
    (driver.getSeqType() == seq_type_t::aa) ? ex::sync_wait(ex::schedule(sch) | ex::bulk(nthreads, aa_kernel)) :
                                              ex::sync_wait(ex::schedule(sch) | ex::bulk(nthreads, dna_kernel));

}