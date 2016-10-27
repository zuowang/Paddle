/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "TrainerInternal.h"

#include <fenv.h>
#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>

#include <google/protobuf/text_format.h>

#include "paddle/math/SufficientVector.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/StringUtil.h"
#include "paddle/utils/Util.h"
#include "paddle/utils/GlobalConstants.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/gserver/layers/ValidationLayer.h"

#include "ThreadParameterUpdater.h"
#include "RemoteParameterUpdater.h"

namespace paddle {

void TrainerInternal::init(const std::shared_ptr<TrainerConfigHelper> &config,
                           const GradientMachinePtr &gradientMachine,
                           std::unique_ptr<TrainerInternalConfig> &&intconfig,
                           const std::shared_ptr<TrainerStats> &stats,
                           bool testing) {
    config_ = config;
    intconfig_ = std::move(intconfig);
    stats_ = stats;

    //! in training will use parameter updater definitly.
    //! But only use parameter in testing mode when some parameter in pserver.
    if (!testing || (config_->getOptConfig().use_sparse_remote_updater() &&
                   intconfig_->loadsave_parameters_in_pserver)) {
      createParameterUpdater(testing);
    }

    gradientMachine_ = gradientMachine;
    if (!gradientMachine) {
      gradientMachine_.reset(GradientMachine::create(
        config_->getConfig().model_config(), intconfig_->mode,
        parameterUpdater_->getParameterTypes()));
    }

    if (FLAGS_use_svb) initCommBus();
}

void TrainerInternal::trainOneBatch(int64_t batchId,
                                    const DataBatch& dataBatch) {
  // true means updating parameter whenever gradient is ready during backward()
  bool doPipelineUpdate =
      (intconfig_->mode != GradientMachine::kSgdSparseCpuTraining) &&
      (intconfig_->local || intconfig_->use_gpu ||
       intconfig_->trainer_count <= 1) && !true;//intconfig_->use_svrg;

  int64_t actualBatchSize = dataBatch.getSize();
  if (actualBatchSize == 0) {
    return;
  }

  bool showStats = intconfig_->show_param_stats_period > 0 &&
                   (batchId + 1) % intconfig_->show_param_stats_period == 0 &&
                   intconfig_->trainer_id == 0;

  std::vector<ParaStat> paraStats;
  if (showStats) {
    paraStats.resize(gradientMachine_->getParameters().size());
  }

  const std::vector<Argument>& inArgs = dataBatch.getStreams();
  std::vector<Argument> outArgs;

  PassType passType = parameterUpdater_->startBatch(actualBatchSize);

  if (config_->getOptConfig().use_sparse_remote_updater()) {
    REGISTER_TIMER("prefetch");
    gradientMachine_->prefetch(inArgs);
    parameterUpdater_->getParametersRemote();
  }

  UpdateCallback updateCallback =
      [this, showStats, &paraStats](Parameter* para) {
    if (showStats) {
      //! @TODO(yuyang18) Show stats is actually a ParameterHook, refactor
      // it
      //! to ParameterHook.
      auto& grad = para->getBuf(PARAMETER_GRADIENT);
      SetDevice device(para->getDeviceId());
      paraStats[para->getID()].avgAbsGrad = grad->getAbsSum() / para->getSize();
      paraStats[para->getID()].maxAbsGrad = grad->getAbsMax();
    }
    if (!para->useSVB()) {
      parameterUpdater_->update(para);
    } else {
      updateSVBParameter(para);
    }
  };

  {
#ifndef PADDLE_DISABLE_TIMER
    Timer timer;
    timer.start();
#endif
    REGISTER_TIMER("forwardBackward");
    forwardBackwardBatch(inArgs, outArgs, passType, updateCallback,
                         doPipelineUpdate);
    if (true) {//intconfig_->use_svrg) {
      std::vector<ParameterPtr>& parameters = gradientMachine_->getParameters();
      for (auto& para : parameters) {
        // copy PARAMETER_GRADIENT to PARAMETER_GRADIENT_TMP
        para->getBuf(PARAMETER_GRADIENT_CUR)->copyFrom(
            *para->getBuf(PARAMETER_GRADIENT));
        // copy PARAMTER_SNAPSHOT to PARAMTER_VALUE
        para->getBuf(PARAMTER_VALUE)->copyFrom(
            *para->getBuf(PARAMTER_SNAPSHOT));
      }

      forwardBackwardBatch(inArgs, outArgs, passType, updateCallback,
                           doPipelineUpdate);
      // PARAMETER_GRADIENT as PARAMETER_GRADIENT_TMP sub PARAMETER_GRADIENT
      for (auto& para : parameters) {
        // PARAMETER_GRADIENT = PARAMETER_GRADIENT_TMP - PARAMETER_GRADIENT
        para->getBuf(PARAMETER_GRADIENT_CUR)->add(
            *para->getBuf(PARAMETER_GRADIENT), -1.0f, 1.0f)
      }
    }
#ifndef PADDLE_DISABLE_TIMER
    timer.stop();
    parameterUpdater_->setForwardbackwardTime(timer.get());
#endif
  }

  if (!doPipelineUpdate) {
    auto& parameters = gradientMachine_->getNonStaticParameters();
    for (auto& para : parameters) {
      updateCallback(para.get());
    }
  }

  real cost = 0;
  {
    REGISTER_TIMER("sumCost");
    cost = Argument::sumCosts(outArgs);
  }

  if (batchId % intconfig_->log_period == 0) {
    currentEvaluator_->start();
    stats_->resetCurrentStat();
  }
  {
    REGISTER_TIMER("eval");
    gradientMachine_->eval(currentEvaluator_);
    gradientMachine_->eval(evaluator_);
  }

  *stats_ += { actualBatchSize, cost };
  {
    REGISTER_TIMER("finishBatch");
    parameterUpdater_->finishBatch(cost);
  }

  if (showStats) {
    showParameterStats(paraStats);
  }
  if ((batchId + 1) % intconfig_->log_period == 0) {
    currentEvaluator_->finish();

    if (intconfig_->dot_period > 0) {
      std::cerr << std::endl;
    }
    LOG(INFO) << " Batch=" << batchId + 1 << " "
              << *stats_
              << " Eval: " << *evaluator_
              << " CurrentEval: " << *currentEvaluator_;
  } else if (intconfig_->dot_period > 0 &&
            (batchId + 1) % intconfig_->dot_period == 0) {
    std::cerr << ".";
  }
}

/**
 * finish train pass
 */
void TrainerInternal::finishTrainPass(int passId, int batchId) {
  gradientMachine_->onPassEnd();
  parameterUpdater_->finishPass();
  evaluator_->finish();
  LOG(INFO) << " Pass=" << passId << " Batch=" << batchId
            << " " << stats_->getStats(false /*without current cost*/)
            << " Eval: " << *evaluator_;
}

void TrainerInternal::showParameterStats(const std::vector<ParaStat>&
                                        paraStats) {
  std::vector<ParameterPtr>& parameters = gradientMachine_->getParameters();
  for (auto& parameter : parameters) {
    SetDevice device(parameter->getDeviceId());
    real sum = parameter->getBuf(PARAMETER_VALUE)->getAbsSum();
    const auto& lr = parameter->getBuf(PARAMETER_LEARNING_RATE);
    std::ostringstream osLrHistogram;
    if (lr) {
      if (VLOG_IS_ON(2)) {
        osLrHistogram << " lr_histogram: ";
        lr->histogram(osLrHistogram);
      } else {
        osLrHistogram << " max_lr=" << std::setw(11) << lr->getMax()
                      << " min_lr=" << std::setw(11) << lr->getMin()
                      << " avg_lr=" << std::setw(11)
                      << lr->getSum() / parameter->getSize();
      }
    }
    int pid = parameter->getID();
    LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
              << std::setw(20) << parameter->getName()
              << " avg_abs_val=" << std::setw(11) << sum / parameter->getSize()
              << " max_val=" << std::setw(11)
              << parameter->getBuf(PARAMETER_VALUE)->getAbsMax()
              << " avg_abs_grad=" << std::setw(11) << paraStats[pid].avgAbsGrad
              << " max_grad=" << std::setw(11) << paraStats[pid].maxAbsGrad
              << osLrHistogram.str();
  }
}

void TrainerInternal::createParameterUpdater(bool testing) {
  const std::string& alg = config_->getOptConfig().algorithm();
  parameterUpdater_.reset(ParameterUpdaterCreators::tryCreateUpdater(
                            alg, config_->getOptConfig(), intconfig_->local,
                            intconfig_->num_passes));
  if (parameterUpdater_) { return; }

  if (!intconfig_->local) {
    if (testing && config_->getOptConfig().use_sparse_remote_updater()) {
      std::unique_ptr<ParameterUpdater> localUpdater;
      localUpdater.reset(
          new SgdLocalUpdater(config_->getOptConfig()));  // do nothing
      parameterUpdater_.reset(new SparseRemoteParameterUpdaterComposite(
          config_->getOptConfig(), intconfig_->num_passes, testing,
          std::move(localUpdater)));
    } else {
      if (GradientMachine::kSgdSparseCpuTraining == intconfig_->mode &&
          !intconfig_->use_old_updater) {
        intconfig_->use_old_updater = true;
        LOG(INFO) << "Sgd sparse training can not work with"
                  << " ConcurrentRemoteParameterUpdater,"
                  << " automatically reset --use_old_updater=true";
      }

      std::unique_ptr<ParameterUpdater> localUpdater;
      if (config_->getOptConfig().num_batches_per_send_parameter() > 1) {
        CHECK(alg == TrainAlgorithm::SGD || alg == TrainAlgorithm::AsyncSGD)
            << "Unsupported algorithm in remote-local mode: " << alg;
        if (GradientMachine::kSgdSparseCpuTraining == intconfig_->mode) {
          localUpdater.reset(new SgdThreadUpdater(*config_));
        } else {
          localUpdater.reset(new SgdLocalUpdater(*config_));
        }
      }

      localUpdater.reset(
              intconfig_->use_old_updater
              ? new RemoteParameterUpdater(
                      *config_,
                      intconfig_->num_passes,
                      std::move(localUpdater))
              : new ConcurrentRemoteParameterUpdater(
                      *config_,
                      intconfig_->num_passes,
                      std::move(localUpdater)));


      if (config_->getOptConfig().use_sparse_remote_updater()) {
        localUpdater.reset(new SparseRemoteParameterUpdaterComposite(
            *config_, intconfig_->num_passes, testing,
            std::move(localUpdater)));
      }

      this->parameterUpdater_ = std::move(localUpdater);

      if (FLAGS_use_svb) {
        if (GradientMachine::kSgdSparseCpuTraining == intconfig_->mode) {
          localParameterUpdater_.reset(new SgdThreadUpdater(*config_));
        } else if (alg == TrainAlgorithm::SGD ||
                   alg == TrainAlgorithm::AsyncSGD) {
          if (config_->getModelConfig().type() == "recursive_nn") {
            localParameterUpdater_.reset(new SgdCpuUpdater(*config_));
          } else if (intconfig_->use_gpu &&
              config_->getOptConfig().do_average_in_cpu() &&
              config_->getOptConfig().average_window() > 0) {
            localParameterUpdater_.reset(
                new SgdUpdaterWithCpuAverager(*config_));
          } else {
            localParameterUpdater_.reset(new SgdLocalUpdater(*config_));
          }
        } else {
          LOG(FATAL) << "Unsupported algorithm in local mode: " << alg;
        }
      }
    }
  } else {
    CHECK_EQ(config_->getOptConfig().num_batches_per_send_parameter(), 1)
        << "num_batches_per_send_parameter should be one in local mode!";

    if (GradientMachine::kSgdSparseCpuTraining == intconfig_->mode) {
      parameterUpdater_.reset(new SgdThreadUpdater(*config_));
    } else if (alg == TrainAlgorithm::SGD || alg == TrainAlgorithm::AsyncSGD) {
      if (config_->getModelConfig().type() == "recursive_nn") {
        parameterUpdater_.reset(new SgdCpuUpdater(*config_));
      } else if (intconfig_->use_gpu &&
                 config_->getOptConfig().do_average_in_cpu() &&
                 config_->getOptConfig().average_window() > 0) {
        parameterUpdater_.reset(
            new SgdUpdaterWithCpuAverager(*config_));
      } else {
        parameterUpdater_.reset(new SgdLocalUpdater(*config_));
      }
    } else {
      LOG(FATAL) << "Unsupported algorithm in local mode: " << alg;
    }
  }
}

void TrainerInternal::forwardBackwardBatch(const std::vector<Argument>& inArgs,
                                   std::vector<Argument>& outArgs,
                                   PassType& passType,
                                   UpdateCallback updateCallback,
                                   bool doPipelineUpdate) {
  gradientMachine_->forwardBackward(
      inArgs, &outArgs, passType, doPipelineUpdate ? updateCallback : nullptr);
}

void TrainerInternal::initCommBus() {
  std::vector<std::string> hosts;
  str::split(FLAGS_trainers, ',', &hosts);
  int trainer_id = FLAGS_trainer_id;
  commBus_.reset(new CommBus(trainer_id, trainer_id, hosts.size()));
  int ltype = (trainer_id == ((int)hosts.size() - 1)) ?
              CommBus::kNone : CommBus::kInterProc;
  CommBus::Config config(trainer_id, ltype, hosts[trainer_id]);
  commBus_->ThreadRegister(config);
  /* This part of the code represents the handshake process to establish
   conenctions to all other trainers. After this part, one trainer can send msgs
   to any other. */

  // trainer i connects to trainer [0, ..., i - 1]
  for (int i = 0; i < trainer_id; ++i) {
    int conn_msg = trainer_id;
    commBus_->ConnectTo(i, hosts[i], &conn_msg, sizeof(conn_msg));
  }

  // trainer i receives connection from trainer [i + 1, n)
  // (n is the total number of trainers);
  // trainer i also receives n messages each from a trainer
  // msgs from trainer [0, i - 1] serves as confirmation of the connection
  // A trainer can broadcast only when it has 1) connected to all lower ones
  // and 2) received confirmation from them
  const int num_expected_conns = hosts.size() - trainer_id - 1;
  int num_conns = 0;
  size_t num_other_msgs = 0;
  const int num_expected_confirms = trainer_id;
  int num_confirms = 0;
  while (num_conns < num_expected_conns
      || num_confirms < num_expected_confirms) {
    zmq::message_t msg;
    int32_t sender_id;
    commBus_->RecvInterProc(&sender_id, &msg);
    int msg_val = *reinterpret_cast<int*>(msg.data());
    if (msg_val == sender_id) {
      num_conns++;
      LOG(INFO) << trainer_id <<  " received conn from " << sender_id
          << " msg = " << msg_val;
    } else {
      num_other_msgs++;
      LOG(INFO) << trainer_id <<  " received nonconn from " << sender_id
          << " msg = " << msg_val;
      if (sender_id < trainer_id) num_confirms++;
    }
  }

  LOG(INFO) << trainer_id << " has received all conns, sending out msgs now";
  for (size_t dst = 0; dst < hosts.size(); dst++) {
    int non_conn_msg = trainer_id + 100;
    if (dst == (size_t)trainer_id) continue;  // cannot send to myself
    commBus_->SendInterProc(dst, &non_conn_msg, sizeof(non_conn_msg));
  }

  LOG(INFO) << trainer_id <<
      " send out all msgs, receive my remaininy msgs now";
  while (num_other_msgs < hosts.size() - 1) {
    zmq::message_t msg;
    int32_t sender_id;
    commBus_->RecvInterProc(&sender_id, &msg);
    int msg_val = *reinterpret_cast<int*>(msg.data());
    if (msg_val == sender_id) {
      LOG(FATAL) << "Error!";
    } else {
      num_other_msgs++;
      LOG(INFO) << trainer_id <<  " received nonconn from " << sender_id
          << " msg = " << msg_val;
    }
  }

  // From now on, one client can send and receive from any other (not itself).
  LOG(INFO) << "Client " << trainer_id << " is connected for SVB";

  max_send_cnt_ = hosts.size();
  max_recv_cnt_ = (hosts.size() - 1) * FLAGS_trainer_count;
}

void TrainerInternal::updateSVBParameter(Parameter* para) {
  LOG(INFO) << "max_recv_cnt_=" << max_recv_cnt_
      << " max_send_cnt_=" << max_send_cnt_;

  while (true) {
    SufficientVector* sv = para->getSV();
    if (sv == nullptr) break;
    // LOG(INFO) << "u height=" << sv->GetU()->getHeight()
    //           << " wight=" << sv->GetU()->getWidth();
    // LOG(INFO) << "v height=" << sv->GetV()->getHeight()
    //           << " wight=" << sv->GetV()->getWidth();

    ProtoSV* psv = new ProtoSV();
    sv->ToProto(psv);
    delete sv;
    std::string msg_str;
    psv->SerializeToString(&msg_str);
    for (int dst = 0; dst < max_send_cnt_; dst++) {
      if (dst == FLAGS_trainer_id) continue;  // cannot send to myself
      size_t sz = commBus_->SendInterProc(dst, msg_str.c_str(), msg_str.size());
      CHECK_EQ(sz, msg_str.size());
      // LOG(INFO) << "Send sv=" << sv << " to " << dst;
    }
    delete psv;
  }

  auto weightGrad = para->getMat(PARAMETER_GRADIENT);
  int recv_cnt = 0;
  while (recv_cnt < max_recv_cnt_) {
    // recv
    zmq::message_t msg;
    int32_t sender_id;
    bool succ = commBus_->RecvInterProcTimeOut(
        &sender_id, &msg, FLAGS_svb_timeout_ms);
    if (!succ) continue;
    // parse
    ProtoSV psv;
    CHECK(psv.ParseFromArray(msg.data(), msg.size())) <<
        "SVB message parsing error\n"
        << msg.size();
    // add to the remote-sv queue
    SufficientVector* sv = new SufficientVector();
    sv->FromProto(psv);
    // LOG(INFO) << "Recv sv=" << sv << " from " << sender_id;
    // LOG(INFO) << "u height=" << sv->GetU()->getHeight()
    //           << " wight=" << sv->GetU()->getWidth();
    // LOG(INFO) << "v height=" << sv->GetV()->getHeight()
    //           << " wight=" << sv->GetV()->getWidth();
    weightGrad->mul(sv->GetU()->getTranspose(), sv->GetV(), 1, 1);
    delete sv;
    ++recv_cnt;
  }

  localParameterUpdater_->update(para);
}

}  // namespace paddle
