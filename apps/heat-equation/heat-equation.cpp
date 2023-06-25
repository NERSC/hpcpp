#include <iostream>

int main(int argc, char *argv[])
{
    return 0;
}

#if 0

//  Copyright (c) 2023 AUTHORS
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <functional>
#include <algorithm>
#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <assert.h>
#include <time.h>

using std::size_t;

const size_t ghosts = 1;
size_t nx = 20;//100000;
const double k = 0.4;
size_t nt = 1;
const double dt = 1.0;
const double dx = 1.0;
size_t threads = 1;

void pr(const std::vector<double>& total) {
  std::cout << "[";
  for(size_t i=0;i<total.size();i++) {
    std::cout << " " << total[i];
  }
  std::cout << " ]" << std::endl;
}

#if 0
class Worker : public std::thread {
public:
  size_t num;
  size_t lo, hi, sz;
  std::vector<double> data, data2;
  static const size_t chsz = 20;
  std::lcos::local::channel_spsc<double> left, right;
  Worker *leftThread = nullptr;
  Worker *rightThread = nullptr;

  void start() {
    std::thread t(std::bind(&Worker::run, this));
    this->swap(t);
  }

  Worker(size_t num_, size_t tx) : num(num_), left(chsz), right(chsz) {
    lo = tx * num;
    hi = tx * (num + 1);
    if (hi > nx) hi = nx;
    lo -= ghosts;
    hi += ghosts;
    sz = hi - lo;
    data.resize(sz);
    data2.resize(sz);

    size_t off=1;
    for(size_t n=0; n < sz; n++) {
      data.at(n) = n + lo + off;
      data2.at(n) = 0.0;
    }
  }
  Worker(Worker&&) noexcept = default;
  ~Worker() {}

  void recv_ghosts() {
    //data[0] = left.get().get();
    //data[sz-1] = right.get().get();
    while(!left.get(&data[0]))
      std::this_thread::yield();
    while(!right.get(&data[sz-1]))
      std::this_thread::yield();
  }

  void update() {
    recv_ghosts();

    for(size_t n=1;n < sz-1;n++)
      data2[n] = data[n] + k*dt/(dx*dx) * (data[n+1] + data[n-1] - 2*data[n]);
    data.swap(data2);

    send_ghosts();
  }

  void send_ghosts() {
    double d1 = data[1];
    double d2 = data[sz-2];
    leftThread->right.set(std::move(d1));
    rightThread->left.set(std::move(d2));
  }

  void run() {
    send_ghosts();
    for(size_t t=0; t < nt; t++)
      update();
    recv_ghosts();
  }
};
#endif

std::vector<double> construct_grid(const std::vector<Worker>& workers) {
  std::vector<double> total;
  total.resize(nx);
  for(const Worker& w : workers) {
    size_t i = w.lo + ghosts;
    for(size_t n = ghosts; n < w.data.size()-ghosts; n++)
      total[i++] = w.data[n];
  }
  return std::move(total);
}

int main(int argc, char **argv) {
  assert(argc == 4);
  threads = std::stoi(argv[1]);
  nt = std::stoi(argv[2]);
  nx = std::stoi(argv[3]);
#if 0
  std::vector<Worker> workers;
  size_t tx = (2*ghosts + nx)/threads;
  for(size_t th=0;th < threads;th++) {
    Worker w(th, tx);
    workers.emplace_back(std::move(w));
  }
  for(size_t th=0;th < threads;th++) {
    size_t next = (th + 1) % threads;
    size_t prev = (threads + th - 1) % threads;
    Worker& w = workers[th];
    w.rightThread = &workers[next];
    w.leftThread = &workers[prev];
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  for(Worker& w : workers) {
    w.start();
  }
  for(Worker& w : workers) {
    w.join();
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -t1).count()*1e-9;
  std::cout << "elapsed: " << elapsed << std::endl;

  std::vector<double> total = construct_grid(workers);
  if(nx <= 20)
    pr(total);
 #endif
  if(!std::filesystem::exists("perfdata.csv")) {
    std::ofstream f("perfdata.csv");
    f << "lang,nx,nt,threads,dt,dx,total time,flops" << std::endl;
    f.close();
  }
  std::ofstream f("perfdata.csv",std::ios_base::app);
  f << "std," << nx << "," << nt << "," << threads << "," << dt << "," << dx << "," << elapsed << ",0" << std::endl;
  f.close();
  return 0;
}
#endif