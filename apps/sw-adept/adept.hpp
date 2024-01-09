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
 * adept driver class
 */

#pragma once

#include "sw-adept.hpp"

using namespace sw;

namespace sw
{

// ------------------------------------------------------------------------------------------------------------------------- //

struct alignments
{
    alignments(int N)
    {
        ref_start = new short[N];
        ref_end = new short[N];
        query_start = new short[N];
        query_end = new short[N];
        topscores = new int[N];
    }

    ~alignments()
    {
        if (ref_start)
        {
            delete[] ref_start;
            ref_start = nullptr;
        }

        if (ref_end)
        {
            delete[] ref_end;
            ref_end = nullptr;
        }

        if (query_start)
        {
            delete[] query_start;
            query_start = nullptr;
        }

        if (query_end)
        {
            delete[] query_end;
            query_end = nullptr;
        }

        if (topscores)
        {
            delete[] topscores;
            topscores = nullptr;
        }
    }

    short *ref_start = nullptr;
    short *ref_end = nullptr;
    short *query_start = nullptr;
    short *query_end = nullptr;
    int   *topscores = nullptr;
};

// ------------------------------------------------------------------------------------------------------------------------- //

class adept
{
private:

    using seq_v = std::vector<string_t>;

    // maximum query length
    int max_query_len;

    // cigar option
    bool cigar;
    seq_type_t seq;

    // batch sizes
    int batch_size;
    int nbatches;
    int N;

    using algn_t = alignments;
    algn_t *algn;

    // set batch size
    void setupArrays(seq_v &_db, seq_v& _q)
    {
        N = _db.size();
        batch_size = std::min(N, MAX_BATCH_SIZE);
        nbatches   = N / batch_size;
        nbatches  += (N % batch_size)? 1 : 0;

        algn = alignments(N);

        if (!this->len_db)
            this->len_db = new int[N+1];
        if (!this->len_queries)
            this->len_queries = new int[N+1];

        for (int k = 0; k < N; k++)
        {
            this->len_db[k] = _db[k].length();
            this->len_queries[k] = _q[k].length();
        }

        // flatten database strings
        auto flattened_str = std::accumulate(_db.begin(), _db.end(), std::string(""));

        if (!this->database)
            this->database = new char[flattened_str.length()];

        // copy to driver.database
        std::memcpy(this->database, flattened_str.c_str(), flattened_str.length());

        flattened_str = std::accumulate(_q.begin(), _q.end(), std::string(""));

        if (!this->queries)
            this->queries = new char[flattened_str.length()];

        // copy to driver.queries
        std::memcpy(this->queries, flattened_str.c_str(), flattened_str.length());

        flattened_str.clear();
    }

public:

    char *database;
    char *queries;
    int *len_db;
    int *len_queries;

    // ------------------------------------------------------------------------------------------------------------------------- //
    adept(seq_type_t _seq, bool _cigar) : cigar(_cigar), seq(_seq)
    {
        // assign blosum62 matrix to scores if amino acids
        (_seq == seq_type_t::aa)? max_query_len = aa::MAX_QUERY_LEN : max_query_len = dna::MAX_QUERY_LEN;
    }

    // ------------------------------------------------------------------------------------------------------------------------- //
    // use default destructor
    ~adept()
    {
        if (algn)
            delete algn;
        if (database)
            delete[] database;
        if (queries)
            delete[] queries;
        if (len_db)
            delete[] len_db;
        if (len_queries)
            delete[] len_queries;
    };

    // ------------------------------------------------------------------------------------------------------------------------- //

    // read and process FASTA files
    int readFASTAs(const string_t &dbFile, const string_t &qFile, std::vector<string_t> &_db, std::vector<string_t> &_q)
    {
        bool status = true;

        // print status for the user here
        std::cout << "\nReading..\ndatabase file: " << dbFile << "\nqueries file: " << qFile << std::endl;

        string_t lineR, lineQ;

        std::ifstream ref_file(dbFile);
        std::ifstream quer_file(qFile);

        // extract reference sequences
        if(ref_file.is_open() && quer_file.is_open())
        {
            int lineNum = 0;

            // get two lines
            while(getline(ref_file, lineR) && getline(quer_file, lineQ))
            {
                lineNum++;

                if(lineR[0] == '>')
                {
                    if (lineQ[0] == '>')
                        continue;
                    else
                    {
                        std::cout << "readFASTAs: mismatch at line: " << lineNum << std::endl;
                        status = false;
                        break;
                    }
                }
                else if (lineR.length() <= MAX_REF_LEN && lineQ.length() <= max_query_len)
                {
                    _db.push_back(lineR);
                    _q.push_back(lineQ);
                }
                else
                    continue;

                if (_db.size() == DATA_SIZE)
                    break;
            }

            // close the files
            ref_file.close();
            quer_file.close();
        }

        if (!status)
            throw std::invalid_argument("FATAL: Invalid database or query files provided.\n");

        // successfully read, set batch size
        this->setupArrays(_db, _q);

        return N;
    }

    // ------------------------------------------------------------------------------------------------------------------------- //

    [[nodiscard]] bool validate(const string_t &vFile)
    {
        // todo: validate here
        return true;
    }

    // ------------------------------------------------------------------------------------------------------------------------- //

    [[nodiscard]] std::array<int, 2> getIndices(int batch_num)
    {
        std::array<int, 2> _indices{-1,-1};

        if (batch_num < nbatches)
        {
            // get start and end pointers
            _indices[0] = batch_num * batch_size;
            _indices[1] = std::min(N, (batch_num+1) * batch_size);
        }

        return _indices;
    }

    // ------------------------------------------------------------------------------------------------------------------------- //

    [[nodiscard]] algn_t *alignments(int N)
    {
        static algn_t *_algn = nullptr;

        if (!_algn)
            _algn = new algn_t(N);

        return _algn;
    }

    // ------------------------------------------------------------------------------------------------------------------------- //

    [[nodiscard]] seq_type_t getSeqType() { return seq; }

    // ------------------------------------------------------------------------------------------------------------------------- //

    [[nodiscard]] bool getCigar() { return cigar; }
};

} // namespace sw