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



#pragma once

#include "ZMQUtil.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/utils/Logging.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {

/*
 * This class resembles a shared bus among all local theads and remote
 * threads.  The goal is to simplify communication handling and unify
 * in-process and network communication under the same interface. Each thread
 * is treated as an unique entity and has its unique global (among all hosts)
 * ID. The ID is the concatnation of the thread's host ID and the thread's
 * local ID.  As it is meant to be simple, it is not doing much error checking
 * and recovery. If something goes wrong, it fails (aborts) quickly to allow
 * debugging to happen immediately.
 * Each thread is an entity and should only register (ThreadRegister) once.
 * A thread is local if it is in the same CommBus object as myself, otherwise it
 * is remote.
 */

class CommBus {
 public:
  static const int kNone = 0;
  static const int kInProc = 1;
  static const int kInterProc = 2;

  struct Config {
   public:
    // My thread id.
    int32_t entity_id_;

    // What should I listen to?
    int ltype_;

    // In the format of "ip:port", such as "192.168.1.1:9999". It must be set
    // if ((ltype_ & kInterProc) == true)
    std::string network_addr_;

    int num_bytes_inproc_send_buff_;
    int num_bytes_inproc_recv_buff_;
    int num_bytes_interproc_send_buff_;
    int num_bytes_interproc_recv_buff_;

    Config():
        entity_id_(0),
        ltype_(kNone),
        num_bytes_inproc_send_buff_(0),
        num_bytes_inproc_recv_buff_(0),
        num_bytes_interproc_send_buff_(0),
        num_bytes_interproc_recv_buff_(0) { }

    Config(int32_t entity_id, int ltype, std::string network_addr):
        entity_id_(entity_id),
        ltype_(ltype),
        network_addr_(network_addr),
        num_bytes_inproc_send_buff_(0),
        num_bytes_inproc_recv_buff_(0),
        num_bytes_interproc_send_buff_(0),
        num_bytes_interproc_recv_buff_(0) { }

    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
  };

  struct ThreadCommInfo {
   public:
    int32_t entity_id_;
    std::unique_ptr<zmq::socket_t> inproc_sock_;
    std::unique_ptr<zmq::socket_t> interproc_sock_;
    // Only contains those listening sockets
    std::vector<std::unique_ptr<zmq::pollitem_t>> pollitems_;
    std::unique_ptr<zmq::pollitem_t> inproc_pollitem_;
    std::unique_ptr<zmq::pollitem_t> interproc_pollitem_;
    int ltype_;
    int32_t poll_size_;

    int num_bytes_inproc_send_buff_;
    int num_bytes_inproc_recv_buff_;
    int num_bytes_interproc_send_buff_;
    int num_bytes_interproc_recv_buff_;

    ThreadCommInfo() { }

    ThreadCommInfo(const ThreadCommInfo&) = delete;
    ThreadCommInfo& operator=(const ThreadCommInfo&) = delete;
  };

  bool IsLocalEntity(int32_t entity_id);

  CommBus(int32_t e_st, int32_t e_end, int32_t num_clients,
      int32_t num_zmq_thrs = 1);
  ~CommBus();
  CommBus(const CommBus&) = delete;
  CommBus& operator=(const CommBus&) = delete;

  // Register a thread, set up necessary commnication channel.
  // For network communication (TCP), zmq::bind may be called after
  // zmq::connect. But for inproc communication, zmq::bind has to be called
  // before zmq::connect.
  // For CommBus, a thread must have successfully registered before
  // other thread may connect to it.
  void ThreadRegister(const Config &config);

  // Connect to a local thread Info is a customer-defined number to be
  // included in the Connect message, how to use it is up to the customer.
  //
  // Comment(wdai): Note that for inproc ThreadRegister must happen before
  // ConnectTo due to a int64_t-standing zeromq bug. See
  // http://grokbase.com/t/zeromq/zeromq-dev/12ajmp3rkd/inproc-need-to-bind-to-an-address-before-connect
  // for more info.
  void ConnectTo(int32_t entity_id, void *connect_msg, size_t size);
  // Connect to a remote thread.
  void ConnectTo(int32_t entity_id, const std::string& network_addr, void
  *connect_msg, size_t size);

  size_t Send(int32_t entity_id, const void *data, size_t len);
  size_t SendInProc(int32_t entity_id, const void *data, size_t len);
  size_t SendInterProc(int32_t entity_id, const void *data, size_t len);

  // msg is nollified
  size_t Send(int32_t entity_id, zmq::message_t &msg);
  size_t SendInProc(int32_t entity_id, zmq::message_t &msg);

  void Recv(int32_t *entity_id, zmq::message_t *msg);
  bool RecvAsync(int32_t *entity_id, zmq::message_t *msg);
  bool RecvTimeOut(int32_t *entity_id, zmq::message_t *msg,
                   int64_t timeout_milli);

  void RecvInProc(int32_t *entity_id, zmq::message_t *msg);
  bool RecvInProcAsync(int32_t *entity_id, zmq::message_t *msg);
  bool RecvInProcTimeOut(int32_t *entity_id, zmq::message_t *msg,
                         int64_t timeout_milli);

  void RecvInterProc(int32_t *entity_id, zmq::message_t *msg);
  bool RecvInterProcAsync(int32_t *entity_id, zmq::message_t *msg);
  bool RecvInterProcTimeOut(int32_t *entity_id, zmq::message_t *msg,
                            int64_t timeout_milli);
  typedef void (CommBus::*RecvFunc)(int32_t *sender_id,
                                    zmq::message_t *zmq_msg);
  typedef bool (CommBus::*RecvTimeOutFunc)(int32_t *sender_id,
                                           zmq::message_t *zmq_msg,
                                           int64_t timeout_milli);
  typedef bool (CommBus::*RecvAsyncFunc)(int32_t *sender_id,
                                         zmq::message_t *msg);

  typedef void (*WaitMsgFunc)(int32_t *sender_id, zmq::message_t *msg);
  typedef bool (*WaitMsgTimeOutFunc)(int32_t *sender_id,
                                     zmq::message_t *msg,
                                     int64_t timeout_milli);

  typedef size_t (CommBus::*SendFunc)(int32_t entity_id, const void *msg,
                                      size_t len);

  SendFunc SendAny_;
  RecvFunc RecvAny_;
  RecvAsyncFunc RecvAsyncAny_;
  RecvTimeOutFunc RecvTimeOutAny_;

 private:
  static void MakeInProcAddr(int32_t entity_id, std::string *result);
  static void MakeInterProcAddr(const std::string &network_addr,
                                std::string *result);

  static void SetUpRouterSocket(zmq::socket_t *sock, int32_t id,
                                int num_bytes_send_buff,
                                int num_bytes_recv_buff);
  static const std::string kInProcPrefix;
  static const std::string kInterProcPrefix;
  zmq::context_t *zmq_ctx_;
  // denote the range of entity IDs that are local, inclusive
  int32_t e_st_;
  int32_t e_end_;
  ThreadLocal<ThreadCommInfo> thr_info_;
};
}  // namespace paddle
