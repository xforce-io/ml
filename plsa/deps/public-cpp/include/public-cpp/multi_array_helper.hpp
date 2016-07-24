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
    inline static void CopyDim2(size_t x, size_t y, const T *const *from, T **to);

    template <typename T>
    inline static void CopyDim3(size_t x, size_t y, size_t z, const T * const* const *from, T ***to);

    template <typename T>
    inline static void SetDim2(size_t x, size_t y, T **to, T v);

    template <typename T>
    inline static void SetDim3(size_t x, size_t y, size_t z, T ***to, T v);

    template <typename T>
    inline static void DeleteDim2(size_t x, size_t y, T **array);

    template <typename T>
    inline static void DeleteDim3(size_t x, size_t y, size_t z, T ***array);
};

template <typename T>
T** MultiArrayHelper::CreateDim2(size_t x, size_t y) {
  T **ret = RCAST<T**>(malloc(x * sizeof(T*)));
  for (size_t i=0; i<x; ++i) {
    ret[i] = RCAST<T*>(malloc(y * sizeof(T)));
  }
  return ret;
}

template <typename T>
T*** MultiArrayHelper::CreateDim3(size_t x, size_t y, size_t z) {
  T ***ret = RCAST<T***>(malloc(x * sizeof(T**)));
  for (size_t i=0; i<x; ++i) {
    ret[i] = RCAST<T**>(malloc(y * sizeof(T*)));
    for (size_t j=0; j<y; ++j) {
      ret[i][j] = RCAST<T*>(malloc(z * sizeof(T)));
    }
  }
  return ret;
}

template <typename T>
void MultiArrayHelper::CopyDim2(size_t x, size_t y, const T *const *from, T **to) {
  for (size_t i=0; i<x; ++i) {
      memcpy(to[i], from[i], y * sizeof(T));
  }
}

template <typename T>
void MultiArrayHelper::CopyDim3(size_t x, size_t y, size_t z, const T *const *const *from, T ***to) {
  for (size_t i=0; i<x; ++i) {
      for (size_t j=0; j<y; ++j) {
          memcpy(to[i][j], from[i][j], z * sizeof(T));
      }
  }
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

template <typename T>
void MultiArrayHelper::DeleteDim2(size_t x, size_t y, T **array) {
  for (size_t i=0; i<x; ++i) {
    free(array[i]);
  }
  free(array);
}

template <typename T>
void MultiArrayHelper::DeleteDim3(size_t x, size_t y, size_t z, T ***array) {
  for (size_t i=0; i<x; ++i) {
    for (size_t j=0; j<y; ++j) {
      free(array[i][j]);
    }
    free(array[i]);
  }
  free(array);
}

}
