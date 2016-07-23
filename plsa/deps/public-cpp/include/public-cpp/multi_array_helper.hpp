#pragma once

#include "common.h"

namespace xforce {

class MultiArrayHelper {
  public:
    template <typename T>
    inline static T** CreateDim2(size_t x, size_t y);

    template <typename T>
    inline static T*** CreateDim3(size_t x, size_t y, size_t z);

    template <typename T>
    inline static void CopyDim2(size_t x, size_t y, const T **from, T **to);

    template <typename T>
    inline static void CopyDim3(size_t x, size_t y, size_t z, const T ***from, T ***to);

    template <typename T>
    inline static void SetDim2(size_t x, size_t y, T **to, T v);

    template <typename T>
    inline static void SetDim3(size_t x, size_t y, size_t z, T ***to, T v);
};

template <typename T>
T** MultiArrayHelper::CreateDim2(size_t x, size_t y) {
  T** ret = new T* [x];
  for (size_t i=0; i<x; ++i) {
    ret[i] = new T [y];
  }
  return ret;
}

template <typename T>
T*** MultiArrayHelper::CreateDim3(size_t x, size_t y, size_t z) {
  T*** ret = new T** [x];
  for (size_t i=0; i<x; ++i) {
    ret[i] = new T* [y];
    for (size_t j=0; j<y; ++j) {
      ret[i][j] = new T [z];
    }
  }
  return ret;
}

template <typename T>
void MultiArrayHelper::CopyDim2(size_t x, size_t y, const T **from, T **to) {
  memcpy(to, from, x * y * sizeof(from[0][0]));
}

template <typename T>
void MultiArrayHelper::CopyDim3(size_t x, size_t y, size_t z, const T ***from, T ***to) {
  memcpy(to, from, x * y * z * sizeof(from[0][0][0]));
}

template <typename T>
void MultiArrayHelper::SetDim2(size_t x, size_t y, T **to, T v) {
  for (size_t i=0; i<x; ++i) {
    for (size_t j=0; j<y; ++j) {
      to[i][j] = v;
    }
  }
}

template <typename T>
void MultiArrayHelper::SetDim3(size_t x, size_t y, size_t z, T ***to, T v) {
  for (size_t i=0; i<x; ++i) {
    for (size_t j=0; j<y; ++j) {
      for (size_t k=0; k<z; ++k)
        to[i][j][k] = v;
    }
  }
}

}
