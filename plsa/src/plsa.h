#include "public.h"

namespace xforce { namespace ml {

class Conf;
class Data;

class Plsa {
  private:
    typedef Plsa Self;

  private:
    static const size_t kNumLocks = 100;

  public:
    Plsa(const Conf &conf, const Data &data);

    void Start();
    void Stop();

  private:
    double CalcRound_();

    void Ready_();
    void CalcEM_();
    double MaxErr_();

    void CalcE_(size_t k, size_t i, size_t j);
    void CalcM1_(size_t j, size_t k);
    void CalcM2_(size_t k, size_t i);
    bool SlaveMove_(size_t &lastStep, size_t &lastPos);
    void SlaveCalc_(size_t lastStep, size_t lastPos);
    inline size_t NumItems_(size_t step);

    inline void LockGuard_(size_t idx);
    inline void UnlockGuard_(size_t idx);

    inline void LockNumItems_();
    inline void UnlockNumItems_();

    bool GetEnd_() const { return end_; }

  private:
    static void* CalcCallback_(void *args);

  private:
    pthread_t *tid_;

    const Conf *conf_;
    const Data *data_;
    size_t numTopics_;
    size_t numDocs_;
    size_t numWords_;

    double ***p_Z_Cond_D_W_;
    double  **p_W_Cond_Z_;
    double  **p_Z_Cond_D_;

    double ***p_Z_Cond_D_W_bak_;
    double  **p_W_Cond_Z_bak_;
    double  **p_Z_Cond_D_bak_;

    SpinLock *guards_;

    size_t step_; // 0=>E, 1=>M1, 2=>M2

    SpinLock lockNumItems_;
    size_t numEStepItems_;
    size_t numM1StepItems_;
    size_t numM2StepItems_;

    bool ***p_Z_Cond_D_W_calc_;
    bool  **p_W_Cond_Z_calc_;
    bool  **p_Z_Cond_D_calc_;

    bool end_;
};

void Plsa::LockGuard_(size_t idx) {
  while (!guards_[idx % kNumLocks].Lock())
    ;
}

void Plsa::UnlockGuard_(size_t idx) {
  guards_[idx % kNumLocks].Unlock();
}

void Plsa::LockNumItems_() {
  while (!lockNumItems_.Lock())
    ;
}

void Plsa::UnlockNumItems_() {
  lockNumItems_.Unlock();
}

size_t Plsa::NumItems_(size_t step) {
  switch (step) {
    case 0 :
      return numTopics_*numDocs_*numWords_;
    case 1 :
      return numWords_*numTopics_;
    default :
      return numTopics_*numDocs_;
  }
}

}}
