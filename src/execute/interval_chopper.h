#pragma once

#include <tuple>
#include <vector>
#include <functional>
#include <cstdint>

#include <iostream>

using std::vector;
using std::tuple;

using interval_t = tuple<uint64_t, uint64_t>;

// Let [b1,e1), [b2,e2) be two intervals.
// These are the ways that they can be organized:
//   b1 <= b2 <= e1 <= e2
//   b1 <= b2 <= e2 <= e1
//   b1 <= e1 <= b2 <= e2  // intersection is zero
//   b2 <= b1 <= e1 <= e2
//   b2 <= b1 <= e2 <= e1
//   b2 <= e2 <= b1 <= e1  // intersection is zero
bool disjoint(interval_t const& i1, interval_t const& i2) {
  // Assuming b1 < e1 and b2 < e2
  auto const& [b1,e1] = i1;
  auto const& [b2,e2] = i2;
  return e1 <= b2 || e2 <= b1;
}

struct interval_chopper_t {
  void increment(interval_t const& interval) {
    do_it(interval, [](int& v){ v++; });
  }

  void decrement(interval_t const& interval) {
    do_it(interval, [](int& v){ v--; });
  }

  bool is_zero(interval_t const& interval) const {
    for(auto const& [beg,end,_]: info) {
      if(!disjoint({beg,end}, interval)) {
        return false;
      }
    }
    return true;
  }

  void print(std::ostream& out) const {
    if(info.size() == 0) {
      out << "chopper [empty]";
    } else {
      out << "chopper";
      for(auto const& [beg,end,cnt]: info) {
        out << " [" << beg << "," << end << ")@" << cnt;
      }
    }
  }

private:
  struct info_t {
    info_t(interval_t const& interval):
      info_t(interval, 0)
    {}

    info_t(interval_t const& interval, int cnt):
      beg(std::get<0>(interval)),
      end(std::get<1>(interval)),
      cnt(cnt)
    {}

    uint64_t beg;
    uint64_t end;
    int      cnt;
  };
  using iter_t = vector<info_t>::iterator;

  // The invariant: info is always sorted,
  //                containing disjoint intervals,
  //                and never contains zero values
  vector<info_t> info;

  tuple<iter_t, iter_t>
  chop(interval_t const& insert_interval)
  {
    // Case 0: info is empty
    if(info.size() == 0) {
      info.emplace_back(insert_interval);
      return {info.begin(), info.begin() + 1};
    }

    auto const& [beg_insert, end_insert] = insert_interval;

    // Case 1: insert_interval is less than everything
    {
      auto const& [beg, end, _] = info.front();
      if(end_insert <= beg) {
        info.insert(info.begin(), info_t(insert_interval));
        return {info.begin(), info.begin()+1};
      }
    }

    // Case 2: insert_interval is greater than everything
    {
      auto const& [beg, end, _] = info.back();
      if(end <= beg_insert) {
        info.emplace_back(insert_interval);
        return {info.end()-1, info.end()};
      }
    }

    // Case 3: insert_interval has some overlap
    vector<info_t> ret;

    int idx_beg_insert, idx_end_insert;
    auto push_back = [&](interval_t const& interval, int cnt) {
      auto const& [beg, end] = interval;
      if(beg == beg_insert) {
        idx_beg_insert = ret.size();
      }
      if(end == end_insert) {
        idx_end_insert = ret.size();
      }
      ret.emplace_back(interval, cnt);
    };

    // Case 3 left: some of insert interval is before info starts
    {
      auto const& [beg, end, _] = info.front();
      if(beg_insert < beg) {
        push_back({beg_insert, beg}, 0);
      }
    }

    // Case 3 middle
    vector<uint64_t> _spots;
    _spots.reserve(3);
    auto push_back_positive = [&](info_t const& i) {
      auto const& [beg,end,cnt] = i;
      _spots.resize(0);
      _spots.push_back(beg);
      if(beg_insert > beg && beg_insert < end) {
        _spots.push_back(beg_insert);
      }
      if(end_insert > beg && end_insert < end) {
        _spots.push_back(end_insert);
      }
      _spots.push_back(end);
      for(int idx = 0; idx != _spots.size()-1; ++idx) {
        push_back({_spots[idx], _spots[idx+1]}, cnt);
      }
    };

    auto push_back_negative = [&](
           uint64_t beg,
           uint64_t end)
    {
      beg = std::max(beg, beg_insert);
      end = std::min(end, end_insert);
      if(beg < end) {
        push_back({beg, end}, 0);
      }
    };

    if(info.size() > 1) {
      for(int idx = 0; idx < info.size()-1; ++idx) {
        push_back_positive(info[idx]);
        auto const& [_0, end_neg, _1] = info[idx];
        auto const& [beg_neg, _2, _3] = info[idx+1];
        push_back_negative(end_neg, beg_neg);
      }
    }
    push_back_positive(info.back());

    // Case 3 right: some of insert interval is after info ends
    {
      auto const& [beg, end, _] = info.back();
      if(end < end_insert) {
        push_back({end, end_insert}, 0);
      }
    }

    info = ret;
    return {info.begin() + idx_beg_insert, info.begin() + idx_end_insert + 1};
  }

  void do_it(interval_t const& interval, std::function<void(int&)> modify) {
    auto [iter, end] = this->chop(interval);
    vector<iter_t> zeros;
    zeros.reserve(std::distance(iter, end));
    for(; iter != end; ++iter) {
      modify(iter->cnt);
      if(iter->cnt == 0) {
        zeros.push_back(iter);
      }
    }
    for(int i = zeros.size() - 1; i >= 0; --i) {
      auto erase_iter = zeros[i];
      info.erase(erase_iter);
    }
  }
};

std::ostream& operator<<(std::ostream& out, interval_chopper_t const& c) {
  c.print(out);
  return out;
}

// void tester(vector<interval_t> xs, vector<interval_t> ys) {
//   interval_chopper_t chopper;
//   for(auto const& i: xs) {
//     chopper.increment(i);
//   }
//   for(auto const& y: ys) {
//     chopper.increment(y);
//     std::cout << chopper << std::endl;
//   }
// }
//
// void test00() {
//   interval_chopper_t chopper;
//   chopper.increment({0, 100});
//   chopper.increment({0, 200});
//   chopper.increment({0, 300});
//   std::cout << chopper << std::endl;
//   chopper.decrement({150, 350});
//   std::cout << chopper << std::endl;
// }
//
// void test01() {
//   vector<interval_t> xs = {{0,10}, {20,30}};
//   tester(xs, {{30,40}});
// }
// void test02() {
//   vector<interval_t> xs = {{0,10}, {20,30}};
//   tester(xs, {{0,10}, {20,30}});
// }
// void test03() {
//   vector<interval_t> xs = {{0,10}, {20,30}};
//   tester(xs, {{10,20}});
// }
// void test04() {
//   vector<interval_t> xs = {{0,10}, {20,30}};
//   tester(xs, {{15,20},{11,13}});
// }
// void test05() {
//   vector<interval_t> xs = {{5,10}, {20,30}};
//   tester(xs, {{5,30}});
// }
// void test06() {
//   vector<interval_t> xs = {{5,10}, {20,30}};
//   tester(xs, {{5,25}});
// }
// void test07() {
//   vector<interval_t> xs = {{5,10}, {20,30}};
//   tester(xs, {{0,35}});
// }
// void test08() {
//   vector<interval_t> xs = {{5,10}, {20,30}};
//   tester(xs, {{0,30}});
// }
//
// int main() {
//   test07();
// }

