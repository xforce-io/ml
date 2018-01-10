#include "public.h"

namespace xforce { namespace ml {

class Conf;
class Data;

class Plsa {
  private:
    typedef Plsa Self;

  private:
    static const size_t kNumLocks = 10000;

  public:
    Plsa(const Conf &conf, const Data &data);

    void Start();
    void Stop();

  private:
    void Init_();
    double CalcRound_();

    void Ready_();
    void CalcEM_();
    double MaxErr_();

    void SetEDominator_();
    inline void CalcE_(size_t k, size_t i, size_t j);
    void SetM1Dominator_();
    inline void CalcM1_(size_t j, size_t k);
    inline void CalcM2_(size_t k, size_t i);
    bool SlaveMove_(size_t id, size_t &lastStep, size_t &lastPos);
    void SlaveCalc_(size_t lastStep, size_t lastPos);

    inline bool LockGuard_(size_t idx);
    inline void UnlockGuard_(size_t idx);

    inline void ClearFinishMark_();
    inline void SetFinishMark_(size_t idx);
    inline bool GetFinishMark_(size_t idx);
    inline bool CheckFinishMark_();

    inline double GetPZCondDW_(size_t i, size_t j, size_t k) const;

    bool GetEnd_() const { return end_; }

    void Dump_();

  private:
    static void* CalcCallback_(void *args);

  private:
    const Conf *conf_;
    const Data *data_;

    size_t numTopics_;
    size_t numDocs_;
    size_t numWords_;

    double **EDominator_;
    double *M1Dominator_;
    double  **p_W_Cond_Z_;
    double  **p_Z_Cond_D_;

    double  **p_W_Cond_Z_bak_;
    double  **p_Z_Cond_D_bak_;

    SpinLock *guards_;

    std::atomic<size_t> step_; // 0=>E, 1=>M1, 2=>M2

    SpinLock lockNumItems_;
    size_t numEStepItems_;
    size_t numM1StepItems_;
    size_t numM2StepItems_;

    bool *p_Z_Cond_D_W_calc_;
    bool *p_W_Cond_Z_calc_;
    bool *p_Z_Cond_D_calc_;

    pthread_t *tid_;
    bool *finishMark_;
    SpinLock lockFinishMark_;

    bool end_;
};

}}

#include "conf.h"
#include "data.h"

namespace xforce { namespace ml {

void Plsa::CalcE_(size_t k, size_t i, size_t j) {
    //double numerator = p_W_Cond_Z_[j][k] * p_Z_Cond_D_[k][i];
    //p_Z_Cond_D_W_[k][i][j] = (numerator > std::numeric_limits<double>::epsilon() &&
            //EDominator_[i][j] > std::numeric_limits<double>::epsilon()) ? numerator/EDominator_[i][j] : 0;
}

void Plsa::CalcM1_(size_t j, size_t k) {
    double numerator = 0.0;
    for (size_t m=0; m<numDocs_; ++m) {
        numerator += data_->GetAccuDocWords(m, j) * GetPZCondDW_(k, m, j);
    }
    p_W_Cond_Z_[j][k] = numerator/M1Dominator_[k];
}

void Plsa::CalcM2_(size_t k, size_t i) {
    double numerator = 0.0;
    for (size_t n=0; n<numWords_; ++n) {
        numerator += data_->GetAccuDocWords(i, n) * GetPZCondDW_(k, i, n);
    }
    p_Z_Cond_D_[k][i] = numerator/data_->GetAccuDocs(i);
}

bool Plsa::LockGuard_(size_t idx) {
    return guards_[idx % kNumLocks].Lock();
}

void Plsa::UnlockGuard_(size_t idx) {
    guards_[idx % kNumLocks].Unlock();
}

void Plsa::ClearFinishMark_() {
    lockFinishMark_.LockUntilSucc();
    for (size_t i=0; i < conf_->GetNumThreads(); ++i) {
        finishMark_[i] = false;
    }
    lockFinishMark_.Unlock();
}

void Plsa::SetFinishMark_(size_t idx) {
    lockFinishMark_.LockUntilSucc();
    finishMark_[idx] = true;
    lockFinishMark_.Unlock();
}

bool Plsa::GetFinishMark_(size_t idx) {
    return finishMark_[idx];
}

bool Plsa::CheckFinishMark_() {
    lockFinishMark_.LockUntilSucc();
    bool ret = true;
    for (size_t i=0; i < conf_->GetNumThreads(); ++i) {
        if (!finishMark_[i]) {
            ret = false;
        }
    }
    lockFinishMark_.Unlock();
    return ret;
}

double Plsa::GetPZCondDW_(size_t k, size_t i, size_t j) const {
    double numerator = p_W_Cond_Z_bak_[j][k] * p_Z_Cond_D_bak_[k][i];
    return (numerator > std::numeric_limits<double>::epsilon() &&
            EDominator_[i][j] > std::numeric_limits<double>::epsilon()) ? numerator/EDominator_[i][j] : 0;
}

}}
