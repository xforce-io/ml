#pragma once

#include "public.h"

namespace xforce { namespace ml {

class Conf {
  public:
    bool Init();  

    size_t GetNumThreads() const { return numThreads_; }
    std::string GetDatapath() const { return datapath_; }
    double GetEpsilon() const { return epsilon_; }
    size_t GetWordsPerDoc() const { return wordsPerDoc_; }

  private:
    size_t numThreads_;  
    std::string datapath_;
    double epsilon_;
    size_t wordsPerDoc_;
};

}}
