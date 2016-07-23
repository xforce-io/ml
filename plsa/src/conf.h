#pragma once

#include "public.h"

namespace xforce { namespace ml {

class Conf {
  public:
    bool Init();  

    size_t GetNumThreads() const { return numThreads_; }

  private:
    size_t numThreads_;  
};

}}
