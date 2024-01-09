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
#include "sw-adept-kernels.hpp"

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
