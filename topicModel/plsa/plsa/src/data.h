#pragma once

#include "public.h"
#include "conf.h"

namespace xforce { namespace ml {

class Data {
  public:
    bool Init(const Conf &conf);

    size_t GetNumTopics() const { return numTopics_; }
    size_t GetNumDocs() const { return numDocs_; }
    size_t GetNumWords() const { return numWords_; }
    inline uint8_t GetAccuDocWords(size_t i, size_t j) const;
    inline size_t GetAccuDocs(size_t i) const;

  private:
    size_t numTopics_;
    size_t numDocs_;
    size_t numWords_;

    uint8_t **accuDocWords_;
    size_t *accuDocs_;
};

uint8_t Data::GetAccuDocWords(size_t i, size_t j) const {
  return accuDocWords_[i][j];
}

size_t Data::GetAccuDocs(size_t i) const {
  return accuDocs_[i];
}

}}
