#include "../plsa.h"
#include "../data.h"

namespace xforce { namespace ml {

Plsa::Plsa(const Conf &conf, const Data &data) :
    conf_(&conf),
    data_(&data),
    numTopics_(data.GetNumTopics()),
    numDocs_(data.GetNumDocs()),
    numWords_(data.GetNumWords()),
    EDominator_(MultiArrayHelper::CreateDim2<double>(numDocs_, numWords_)),
    M1Dominator_(new double [numTopics_]),
    p_W_Cond_Z_(MultiArrayHelper::CreateDim2<double>(numWords_, numTopics_)),
    p_Z_Cond_D_(MultiArrayHelper::CreateDim2<double>(numTopics_, numDocs_)), 
    p_W_Cond_Z_bak_(MultiArrayHelper::CreateDim2<double>(numWords_, numTopics_)),
    p_Z_Cond_D_bak_(MultiArrayHelper::CreateDim2<double>(numTopics_, numDocs_)),
    guards_(new SpinLock [kNumLocks]),
    step_(0),
    numEStepItems_(0),
    numM1StepItems_(0),
    numM2StepItems_(0),
    p_Z_Cond_D_W_calc_(new bool [numTopics_]),
    p_W_Cond_Z_calc_(new bool [numTopics_]),
    p_Z_Cond_D_calc_(new bool [numTopics_]),
    tid_(new pthread_t [conf.GetNumThreads()]),
    finishMark_(new bool [conf.GetNumThreads()]),
    end_(false) {}

void Plsa::Start() {
    printf("numTopics[%lu]\n", numTopics_);
    printf("numDocs[%lu]\n", numDocs_);
    printf("numWords[%lu]\n", numWords_);

    Init_();

    for (size_t i=0; i < conf_->GetNumThreads(); ++i) {
        auto para = new std::pair<size_t, Plsa*>(i, this);
        int ret = pthread_create(&(tid_[i]), NULL, CalcCallback_, para);
        if (0 != ret) {
            FATAL("fail_start_thread");
            return;
        }
    }

    size_t round = 0;
    while (true) {
        std::cout << "start_round[" << round << "]" << std::endl;
        double err = CalcRound_();
        std::cout << "round[" << round << "] err[" << err << "]" << std::endl;
        if (err < conf_->GetEpsilon()) {
            break;
        }
        round += 1;
    }
    Dump_();
}

void Plsa::Stop() {
    end_ = true;
    for (size_t i=0; i < conf_->GetNumThreads(); ++i) {
        pthread_join(tid_[i], NULL);
    }
}

void Plsa::Init_() {
    static unsigned int randSeed = 0;
 
    //init p_W_Cond_Z_
    double *wzNormalize = new double [numTopics_];
    for (size_t k=0; k<numTopics_; ++k) {
        wzNormalize[k] = 0.0;
        for (size_t j=0; j<numWords_; ++j) {
            p_W_Cond_Z_[j][k] = rand_r(&randSeed);
            wzNormalize[k] += p_W_Cond_Z_[j][k];
        }
    }

    for (size_t k=0; k<numTopics_; ++k) {
        for (size_t j=0; j<numWords_; ++j) {
            p_W_Cond_Z_[j][k] /= wzNormalize[k];
        }
    }
    delete [] wzNormalize;

    //init p_Z_Cond_D_
    double *zdNormalize = new double [numDocs_];
    for (size_t i=0; i<numDocs_; ++i) {
        zdNormalize[i] = 0.0;
        for (size_t k=0; k<numTopics_; ++k) {
            p_Z_Cond_D_[k][i] = rand_r(&randSeed);
            zdNormalize[i] += p_Z_Cond_D_[k][i];
        }
    }

    for (size_t i=0; i<numDocs_; ++i) {
        for (size_t k=0; k<numTopics_; ++k) {
            p_Z_Cond_D_[k][i] /= zdNormalize[i];
        }
    }
    delete [] zdNormalize;

    numEStepItems_ = numTopics_;
    numM1StepItems_ = numTopics_;
    numM2StepItems_ = numTopics_;

    for (size_t i=0; i < conf_->GetNumThreads(); ++i) {
        SetFinishMark_(i);
    }
}

double Plsa::CalcRound_() {
    Timer t;

    t.Start();
    Ready_();
    t.Stop();
    std::cout << "ready_cost[" << t.TimeMs() << "]" << std::endl;

    t.Start();
    CalcEM_();
    t.Stop();
    std::cout << "em_cost[" << t.TimeMs() << "]" << std::endl;
    return MaxErr_();
}

void Plsa::Ready_() {
    MultiArrayHelper::CopyDim2<double>(numWords_, numTopics_, p_W_Cond_Z_, p_W_Cond_Z_bak_);  
    MultiArrayHelper::CopyDim2<double>(numTopics_, numDocs_, p_Z_Cond_D_, p_Z_Cond_D_bak_);  

    lockNumItems_.LockUntilSucc();
    numEStepItems_=0;
    numM1StepItems_=0;
    numM2StepItems_=0;
    lockNumItems_.Unlock();

    for (size_t k=0; k<numTopics_; ++k) {
        p_Z_Cond_D_W_calc_[k] = false;
    }
    for (size_t k=0; k<numTopics_; ++k) {
        p_W_Cond_Z_calc_[k] = false;
    }
    for (size_t k=0; k<numTopics_; ++k) {
        p_Z_Cond_D_calc_[k] = false;
    }

    SetEDominator_();
    step_ = 0;
    ClearFinishMark_();
}

void Plsa::CalcEM_() {
    Timer t;
    while (true) {
        switch (step_) {
            case 0 :
                if (numEStepItems_ != numTopics_ || !CheckFinishMark_()) {
                    usleep(10);
                    break;
                } else {
                    t.Stop();
                    std::cout << "E_cost[" << t.TimeMs() << "]" << std::endl;
                    t.Start();

                    SetM1Dominator_();
                    step_ = 1;
                    ClearFinishMark_();
                }
                break;
            case 1 :
                if (numM1StepItems_ != numTopics_ || !CheckFinishMark_()) {
                    usleep(10);
                    break;
                } else {
                    t.Stop();
                    std::cout << "M1_cost[" << t.TimeMs() << "]" << std::endl;
                    t.Start();

                    step_ = 2;
                    ClearFinishMark_();
                }
                break;
            default :
                if (numM2StepItems_ != numTopics_ || !CheckFinishMark_()) {
                    usleep(10);
                    break;
                } else {
                    t.Stop();
                    std::cout << "M2_cost[" << t.TimeMs() << "]" << std::endl;
                    t.Start();
                    return;
                }
                break;
        }
    }
}

double Plsa::MaxErr_() {
    size_t maxIdxK=0, maxIdxI=0, maxIdxJ=0, maxStep=0;
    double maxErr=0;
    for (size_t j=0; j<numWords_; ++j) {
        for (size_t k=0; k<numTopics_; ++k) {
            double err = fabs(p_W_Cond_Z_[j][k] - p_W_Cond_Z_bak_[j][k]);
            if (err > maxErr) {
                maxIdxJ = j;
                maxIdxK = k;
                maxStep = 1;

                maxErr = err;
            }
        }
    }

    double sum=0;
    for (size_t k=0; k<numTopics_; ++k) {
        for (size_t i=0; i<numDocs_; ++i) {
            sum += p_Z_Cond_D_[k][i];

            double err = fabs(p_Z_Cond_D_[k][i] - p_Z_Cond_D_bak_[k][i]);
            if (err > maxErr) {
                maxIdxK = k;
                maxIdxI = i;
                maxStep = 2;

                maxErr = err;
            }
        }
    }
    printf("sum[%.9f] max_kij[%lu|%lu|%lu] max_step[%lu]\n", sum, maxIdxK, maxIdxI, maxIdxJ, maxStep);
    return maxErr;
}

void Plsa::SetEDominator_() {
    for (size_t i=0; i<numDocs_; ++i) {
        for (size_t j=0; j<numWords_; ++j) {
            double dominator = 0.0;
            for (size_t l=0; l<numTopics_; ++l) {
                dominator += p_W_Cond_Z_[j][l] * p_Z_Cond_D_[l][i];
            }
            EDominator_[i][j] = dominator;
        }
    }
}

void Plsa::SetM1Dominator_() {
    for (size_t k=0; k<numTopics_; ++k) {
        double dominator=0.0;
        for (size_t m=0; m<numDocs_; ++m) {
            for (size_t n=0; n<numWords_; ++n) {
                dominator += data_->GetAccuDocWords(m, n) * GetPZCondDW_(k, m, n);
            }
        }
        M1Dominator_[k] = dominator;
    }
}

bool Plsa::SlaveMove_(size_t id, size_t &lastStep, size_t &lastPos) {
    if (GetFinishMark_(id)) {
        return false;
    }

    //check status
    if (step_ != lastStep) {
        lastStep = step_;

        static unsigned int kRandSeed = 0;
        lastPos = rand_r(&kRandSeed);
    } else {
        lastPos=lastPos+1;
    }

    switch (lastStep) {
        case 0 :
            if (numEStepItems_ == numTopics_) {
                SetFinishMark_(id);
                return false;
            }
            break;
        case 1 :
            if (numM1StepItems_ == numTopics_) {
                SetFinishMark_(id);
                return false;
            }
            break;
        default :
            if (numM2StepItems_ == numTopics_) {
                SetFinishMark_(id);
                return false;
            }
            break;
    }
    lastPos = lastPos % numTopics_;
    return true;
}

void Plsa::SlaveCalc_(size_t lastStep, size_t lastPos) {
    size_t numItems = numTopics_;
    switch (lastStep) {
        case 0 :
            if (numEStepItems_ != numItems) {
                size_t x = lastPos;
                if (LockGuard_(x)) {
                    if (!p_Z_Cond_D_W_calc_[x]) {
                        for (size_t y=0; y<numDocs_; ++y) {
                            for (size_t z=0; z<numWords_; ++z) {
                                CalcE_(x, y, z);
                            }
                        }
                        p_Z_Cond_D_W_calc_[x] = true;

                        lockNumItems_.LockUntilSucc();
                        if (numEStepItems_ != numItems) {
                            ++numEStepItems_;
                        } else {
                            assert(false);
                        }
                        lockNumItems_.Unlock();
                    }
                    UnlockGuard_(x);
                }
                return;
            }
            break;
        case 1 :
            if (numM1StepItems_ != numItems) {
                size_t y = lastPos;
                if (LockGuard_(y)) {
                    if (!p_W_Cond_Z_calc_[y]) {
                        for (size_t x=0; x<numWords_; ++x) {
                            CalcM1_(x, y);
                        }
                        p_W_Cond_Z_calc_[y] = true;

                        lockNumItems_.LockUntilSucc();
                        if (numM1StepItems_ != numItems) {
                            ++numM1StepItems_;
                        } else {
                            assert(false);
                        }
                        lockNumItems_.Unlock();
                    }
                    UnlockGuard_(y);
                }
                return;
            }
            break;
        default :
            if (numM2StepItems_ != numItems) {
                size_t x = lastPos;
                if (LockGuard_(x)) {
                    if (!p_Z_Cond_D_calc_[x]) {
                        for (size_t y=0; y<numDocs_; ++y) {
                            CalcM2_(x, y);
                        }
                        p_Z_Cond_D_calc_[x] = true;

                        lockNumItems_.LockUntilSucc();
                        if (numM2StepItems_ != numItems) {
                            ++numM2StepItems_;
                        } else {
                            assert(false);
                        }
                        lockNumItems_.Unlock();
                    }
                    UnlockGuard_(x);
                }
                return;
            }
            break;
    }
}

void* Plsa::CalcCallback_(void *args) {
    std::pair<size_t, Plsa*> *para = RCAST<std::pair<size_t, Plsa*>*>(args);
    size_t id = para->first;
    Self* self = para->second;
    delete para;

    size_t lastStep = 2;
    size_t lastPos = 2;
    while (!self->GetEnd_()) {
        if (!self->SlaveMove_(id, lastStep, lastPos)) {
            usleep(10);
            continue;
        }
        self->SlaveCalc_(lastStep, lastPos);
    }
    return NULL;
}

void Plsa::Dump_() {
    for (size_t k=0; k<numTopics_; ++k) {
        for (size_t j=0; j<numWords_; ++j) {
            if (p_W_Cond_Z_[j][k] > std::numeric_limits<double>::epsilon()) {
                printf("jk=%lu=%lu=%.9f\n", j, k, p_W_Cond_Z_[j][k]);
            }
        }
    }

    for (size_t k=0; k<numTopics_; ++k) {
        for (size_t i=0; i<numDocs_; ++i) {
            if (p_Z_Cond_D_[k][i] > std::numeric_limits<double>::epsilon()) {
                printf("ki=%lu=%lu=%.9f\n", k, i, p_Z_Cond_D_[k][i]);
            }
        }
    }
}

}}
