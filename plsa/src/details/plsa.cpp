#include "../plsa.h"
#include "../conf.h"
#include "../data.h"

namespace xforce { namespace ml {

Plsa::Plsa(const Conf &conf, const Data &data) :
  conf_(&conf),
  data_(&data),
  numTopics_(data.GetNumTopics()),
  numDocs_(data.GetNumDocs()),
  numWords_(data.GetNumWords()),
  p_Z_Cond_D_W_(MultiArrayHelper::CreateDim3<double>(numTopics_, numDocs_, numWords_)),
  p_W_Cond_Z_(MultiArrayHelper::CreateDim2<double>(numWords_, numTopics_)),
  p_Z_Cond_D_(MultiArrayHelper::CreateDim2<double>(numTopics_, numDocs_)), 
  p_Z_Cond_D_W_bak_(MultiArrayHelper::CreateDim3<double>(numTopics_, numDocs_, numWords_)),
  p_W_Cond_Z_bak_(MultiArrayHelper::CreateDim2<double>(numWords_, numTopics_)),
  p_Z_Cond_D_bak_(MultiArrayHelper::CreateDim2<double>(numTopics_, numDocs_)),
  guards_(new SpinLock [kNumLocks]),
  step_(0),
  numEStepItems_(0),
  numM1StepItems_(0),
  numM2StepItems_(0),
  p_Z_Cond_D_W_calc_(MultiArrayHelper::CreateDim3<bool>(numTopics_, numDocs_, numWords_)),
  p_W_Cond_Z_calc_(MultiArrayHelper::CreateDim2<bool>(numWords_, numTopics_)),
  p_Z_Cond_D_calc_(MultiArrayHelper::CreateDim2<bool>(numTopics_, numDocs_)),
  end_(false) {}

void Plsa::Start() {
  tid_ = new pthread_t [conf_->GetNumThreads()];
  for (size_t i=0; i < conf_->GetNumThreads(); ++i) {
    int ret = pthread_create(&(tid_[i]), NULL, CalcCallback_, this);
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
    round += 1;
  }
}

void Plsa::Stop() {
  end_ = true;
  for (size_t i=0; i < conf_->GetNumThreads(); ++i) {
    pthread_join(tid_[i], NULL);
  }
}

double Plsa::CalcRound_() {
  Ready_();
  CalcEM_();
  return MaxErr_();
}

void Plsa::Ready_() {
  MultiArrayHelper::CopyDim3<double>(numTopics_, numDocs_, numWords_, p_Z_Cond_D_W_, p_Z_Cond_D_W_bak_);  
  MultiArrayHelper::CopyDim2<double>(numWords_, numTopics_, p_W_Cond_Z_, p_W_Cond_Z_bak_);  
  MultiArrayHelper::CopyDim2<double>(numTopics_, numDocs_, p_Z_Cond_D_, p_Z_Cond_D_bak_);  

  LockNumItems_();
  numEStepItems_=0;
  numM1StepItems_=0;
  numM2StepItems_=0;
  UnlockNumItems_();
  
  MultiArrayHelper::SetDim3<bool>(numTopics_, numDocs_, numWords_, p_Z_Cond_D_W_calc_, false);
  MultiArrayHelper::SetDim2<bool>(numWords_, numTopics_, p_W_Cond_Z_calc_, false);  
  MultiArrayHelper::SetDim2<bool>(numTopics_, numDocs_, p_Z_Cond_D_calc_, false);  

  step_ = 0;
}

void Plsa::CalcEM_() {
  while (true) {
    switch (step_) {
      case 0 :
        if (numEStepItems_ != NumItems_(step_)) {
          usleep(10000);
          break;
        } else {
          step_ = 1;
        }
        break;
      case 1 :
        if (numM1StepItems_ != NumItems_(step_)) {
          usleep(10000);
          break;
        } else {
          step_ = 2;
        }
        break;
      default :
        if (numM2StepItems_ != NumItems_(step_)) {
          usleep(10000);
          break;
        } else {
          return;
        }
        break;
    }
  }
}

double Plsa::MaxErr_() {
  double maxErr=0;
  for (size_t k=0; k<numTopics_; ++k) {
    for (size_t i=0; i<numDocs_; ++i) {
      for (size_t j=0; j<numWords_; ++j) {
        double err = abs(p_Z_Cond_D_W_[k][i][j] - p_Z_Cond_D_W_bak_[k][i][j]);
        if (err > maxErr) {
          maxErr = err;
        }
      }
    }
  }

  for (size_t j=0; j<numWords_; ++j) {
    for (size_t k=0; k<numTopics_; ++k) {
      double err = abs(p_W_Cond_Z_[j][k] - p_W_Cond_Z_bak_[j][k]);
      if (err > maxErr) {
        maxErr = err;
      }
    }
  }

  for (size_t k=0; k<numTopics_; ++k) {
    for (size_t i=0; i<numDocs_; ++i) {
      double err = abs(p_Z_Cond_D_[k][i] - p_Z_Cond_D_bak_[k][i]);
      if (err > maxErr) {
        maxErr = err;
      }
    }
  }
  return maxErr;
}

void Plsa::CalcE_(size_t k, size_t i, size_t j) {
  double numerator = p_W_Cond_Z_[j][k] * p_Z_Cond_D_[k][i];
  double dominator = 0.0;
  for (size_t l=0; l<numTopics_; ++l) {
    dominator += p_W_Cond_Z_[j][l] * p_Z_Cond_D_[l][i];
  }
  p_Z_Cond_D_W_[k][i][j] = numerator/dominator;
}

void Plsa::CalcM1_(size_t j, size_t k) {
  double numerator = 0.0;
  double dominator = 0.0;
  for (size_t m=0; m<numDocs_; ++m) {
    numerator += data_->GetAccuDocWords(m, j) * p_Z_Cond_D_W_[k][m][j];
  }

  for (size_t m=0; m<numDocs_; ++m) {
    for (size_t n=0; n<numWords_; ++n) {
      dominator += data_->GetAccuDocWords(m, n) * p_Z_Cond_D_W_[k][m][n];
    }
  }
  p_W_Cond_Z_[j][k] = numerator/dominator;
}

void Plsa::CalcM2_(size_t k, size_t i) {
  double numerator = 0.0;
  double dominator = data_->GetAccuDocs(i);
  for (size_t n=0; n<numWords_; ++n) {
    numerator += data_->GetAccuDocWords(i, n) * p_Z_Cond_D_W_[k][i][n];
  }
  p_Z_Cond_D_[k][i] = numerator/dominator;
}

bool Plsa::SlaveMove_(size_t &lastStep, size_t &lastPos) {
  //check status
  if (step_!=lastStep) {
    lastStep = step_;

    static unsigned int kRandSeed = 0;
    lastPos = rand_r(&kRandSeed);
  } else {
    lastPos=lastPos+1;
    size_t numItems = NumItems_(lastStep);
    switch (lastStep) {
      case 0 :
        if (numEStepItems_ == numItems) {
          return false;
        }
        lastPos = lastPos % numItems;
        break;
      case 1 :
        if (numM1StepItems_ == numItems) {
          return false;
        }
        lastPos = lastPos % numItems;
        break;
      default :
        if (numM2StepItems_ == numItems) {
          return false;
        }
        lastPos = lastPos % numItems;
        break;
    }
  }
  return true;
}

void Plsa::SlaveCalc_(size_t lastStep, size_t lastPos) {
  size_t numItems = NumItems_(lastStep);
  switch (lastStep) {
    case 0 :
      if (numEStepItems_ == numItems) {
        size_t x = lastPos / (numDocs_*numWords_);
        size_t rest = lastPos % (numDocs_*numWords_);
        size_t y = rest / numWords_;
        size_t z = rest % numWords_;
        LockGuard_(x);
        if (!p_Z_Cond_D_W_calc_[x][y][z]) {
          CalcE_(x, y, z);
          p_Z_Cond_D_W_calc_[x][y][z] = true;
        }

        LockNumItems_();
        if (numEStepItems_ != numItems) {
          numEStepItems_ += 1;
        }
        UnlockNumItems_();

        UnlockGuard_(x);
        return;
      }
      break;
    case 1 :
      if (numM1StepItems_ == numItems) {
        size_t x = lastPos / numTopics_;
        size_t y = lastPos % numTopics_;
        LockGuard_(x);
        if (!p_W_Cond_Z_calc_[x][y]) {
          CalcM1_(x, y);
          p_W_Cond_Z_calc_[x][y] = true;
        }

        LockNumItems_();
        if (numM1StepItems_ != numItems) {
          numM1StepItems_ += 1;
        }
        UnlockNumItems_();

        UnlockGuard_(x);
        return;
      }
      break;
    default :
      if (numM2StepItems_ == numItems) {
        size_t x = lastPos / numDocs_;
        size_t y = lastPos % numDocs_;
        LockGuard_(x);
        if (!p_Z_Cond_D_calc_[x][y]) {
          CalcM2_(x, y);
          p_Z_Cond_D_calc_[x][y] = true;
        }

        LockNumItems_();
        if (numM2StepItems_ != numItems) {
          numM2StepItems_ += 1;
        }
        UnlockNumItems_();

        UnlockGuard_(x);
        return;
      }
      break;
  }
}

void* Plsa::CalcCallback_(void *args) {
  Self* self = RCAST<Self*>(args);
  size_t lastStep = 0;
  size_t lastPos = 0;
  while (!self->GetEnd_()) {
    if (!self->SlaveMove_(lastStep, lastPos)) {
      usleep(1000);
    }
    self->SlaveCalc_(lastStep, lastPos);
  }
  return NULL;
}

}}
