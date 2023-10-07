/*
 * MIT License
 *
 * Copyright (c) 2023 The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of any
 * required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

//
// counting_iterator taken from
// https://github.com/LLNL/LULESH/blob/2.0.2-dev/stdpar/src/lulesh.h#L687
//

#pragma once

#include "commons.hpp"

using Index_t = int32_t;

struct counting_iterator {
 private:
  using self = counting_iterator;

 public:
  using value_type = Index_t;
  using difference_type = typename std::make_signed<Index_t>::type;
  using pointer = Index_t*;
  using reference = Index_t&;
  using iterator_category = std::random_access_iterator_tag;

  counting_iterator() : value(0) {}

  explicit counting_iterator(value_type v) : value(v) {}

  value_type operator*() const { return value; }

  value_type operator[](difference_type n) const { return value + n; }

  self& operator++() {
    ++value;
    return *this;
  }

  self operator++(int) {
    self result{value};
    ++value;
    return result;
  }

  self& operator--() {
    --value;
    return *this;
  }

  self operator--(int) {
    self result{value};
    --value;
    return result;
  }

  self& operator+=(difference_type n) {
    value += n;
    return *this;
  }

  self& operator-=(difference_type n) {
    value -= n;
    return *this;
  }

  friend self operator+(self const& i, difference_type n) {
    return self(i.value + n);
  }

  friend self operator+(difference_type n, self const& i) {
    return self(i.value + n);
  }

  friend difference_type operator-(self const& x, self const& y) {
    return x.value - y.value;
  }

  friend self operator-(self const& i, difference_type n) {
    return self(i.value - n);
  }

  friend bool operator==(self const& x, self const& y) {
    return x.value == y.value;
  }

  friend bool operator!=(self const& x, self const& y) {
    return x.value != y.value;
  }

  friend bool operator<(self const& x, self const& y) {
    return x.value < y.value;
  }

  friend bool operator<=(self const& x, self const& y) {
    return x.value <= y.value;
  }

  friend bool operator>(self const& x, self const& y) {
    return x.value > y.value;
  }

  friend bool operator>=(self const& x, self const& y) {
    return x.value >= y.value;
  }

 private:
  value_type value;
};