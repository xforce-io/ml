#pragma once

#include "common.h"

namespace xforce { namespace ml {

template <typename T>
class MultiSparseArray3 {
  private:
    typedef CloseHashmap<uint32_t, T> BasicContainer;

  public:
    MultiSparseArray3(uint32_t x, uint32_t y, uint32_t z_estimate);

    void Set(uint32_t x, uint32_t y, uint32_t z, T v)
    T Get(uint32_t x, uint32_t y, uint32_t z);

    virtual ~MultiSparseArray3();

  private:  
    uint32_t x_;
    uint32_t y_;
    BasicContainer **basicContainer_;
};

template <typename T>
MultiSparseArray3<T>::MultiSparseArray3(uint32_t x, uint32_t y, uint32_t z_estimate) {
  x_ = x;
  y_ = y;
  basicContainer_ = new BasicContainer* [x];
  for (uint32_t i=0; i<x; ++i) {
    basicContainer_[i] = new BasicContainer(z_estimate * kLoadFactor) [y];
  }
}

template <typename T>
void MultiSparseArray3<T>::Set(uint32_t x, uint32_t y, uint32_t z, const T &v) {
  if (v < std::numeric_limits<double>::epsilon()) {
    basicContainer_[x][y].Erase(z);
  } else {
    basicContainer_[x][y].Upsert(z, v);
  }
}

template <typename T>
T MultiSparseArray3<T>::Get(uint32_t x, uint32_t y, uint32_t z) {
  const T* value = basicContainer_[x][y].Get(z);
  if (value != NULL) {
    return *value;
  } else {
    return 0;
  }
}

template <typename T>
MultiSparseArray3<T>::~MultiSparseArray3() {
  for (uint32_t i=0; i<x_; ++i) {
    delete [] basicContainer_[i];
  }
  delete [] basicContainer_;
}

}}
