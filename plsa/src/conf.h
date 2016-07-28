#pragma once

#include "public.h"

namespace xforce { namespace ml {

class Conf {
  public:
    bool Init();  

    size_t GetNumThreads() const { return numThreads_; }
    std::string GetDatapath() const { return datapath_; }
    double GetEpsilon() const { return epsilon_; }

  private:
    size_t numThreads_;  
    std::string datapath_;
    double epsilon_;
};

}}
