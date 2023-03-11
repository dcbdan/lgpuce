#pragma once

#include "interval_chopper.h"

#include <mutex>
#include <condition_variable>

// Here is the idea:
//   at all times, store all <reading> intervals and <writing> intervals
struct interval_lock_t {
  // store [begin,end) indices

  struct raii_lock_t {
    raii_lock_t(vector<interval_t> const& read,
                vector<interval_t> const& write,
                interval_lock_t* self):
      read(read), write(write), self(self)
    {
      self->lock(this->read, this->write);
    }

    ~raii_lock_t() {
      self->unlock(this->read, this->write);
    }
  private:
    vector<interval_t> read;
    vector<interval_t> write;
    interval_lock_t* self;
  };

  raii_lock_t acquire(vector<interval_t> const& read, vector<interval_t> const& write) {
    return raii_lock_t(read, write, this);
  }

  raii_lock_t acquire_and_correct(
    vector<interval_t>& read, vector<interval_t>& write)
  {
    // TODO:
    // It should be the case that all read and write intervals are disjoint.
    // However, the read and write poritions of an apply op need not be disjoint.
    // For instance, x := x + y would have x as read, x as write and y as read.
    // For the purposes of locks, locking x and both read and write would
    // break everything. Instead, it should just be x write, y read
    //
    // This functions should modify the read and write vectors so that
    // any regions used in both write and read only ends up in write
    // and so that all the intervals are disjoint.
    return raii_lock_t(read, write, this);
  }

  void lock(vector<interval_t> const& read,
            vector<interval_t> const& write)
  {
    // Acquire the lock when the condition is fully satisfied
    std::unique_lock lk(m);
    cv.wait(lk, [this, &read, &write]() { return
      this->available(
        read,
        write);
    });

    for(interval_t const& r: read) {
      reading.increment(r);
    }
    for(interval_t const& w: write) {
      writing.increment(w);
    }
  }

  void unlock(vector<interval_t> const& read,
              vector<interval_t> const& write)
  {
    // Let self know that the locks are no longer necessary
    std::unique_lock lk(m);

    for(interval_t const& r: read) {
      reading.decrement(r);
    }
    for(interval_t const& w: write) {
      writing.decrement(w);
    }
  }

  void print(std::ostream& out) const {
    out << "reading.";
    reading.print(out);
    out << " writing.";
    writing.print(out);
  }

private:
  interval_chopper_t reading;
  interval_chopper_t writing;

  std::mutex m;
  std::condition_variable cv;

  bool available(vector<interval_t> const& read,
                 vector<interval_t> const& write) const
  {
    // All reading intervals are fine as long as they aren't
    // being written to
    for(interval_t const& r: read) {
      if(!writing.is_zero(r)) {
        return false;
      }
    }
    // For writing intervals, they also can't be being read
    for(interval_t const& w: write) {
      if(!(reading.is_zero(w) && writing.is_zero(w))) {
        return false;
      }
    }
    return true;
  }
};

std::ostream& operator<<(std::ostream& out, interval_lock_t const& c) {
  c.print(out);
  return out;
}


