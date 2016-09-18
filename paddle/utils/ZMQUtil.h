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

#include <zmq.hpp>
#include <assert.h>
#include <stdint.h>
#include <time.h>

namespace paddle {

class ZMQUtil {
 public:
  static int32_t EntityID2ZmqID(int32_t entity_id);

  static int32_t ZmqID2EntityID(int32_t zmq_id);

  static void ZMQSetSockOpt(zmq::socket_t *sock, int option,
                            const void *optval, size_t optval_size);

  static void ZMQBind(zmq::socket_t *sock, const std::string &connect_addr);

  static void ZMQConnectSend(zmq::socket_t *sock,
                             const std::string &connect_addr,
                             int32_t zmq_id, void *msg, size_t size);

  // True for received, false for not
  static bool ZMQRecvAsync(zmq::socket_t *sock, zmq::message_t *msg);

  static bool ZMQRecvAsync(zmq::socket_t *sock, int32_t *zmq_id,
                           zmq::message_t *msg);

  static void ZMQRecv(zmq::socket_t *sock, zmq::message_t *msg);

  static void ZMQRecv(zmq::socket_t *sock, int32_t *zmq_id,
                      zmq::message_t *msg);

  /*
   * return number of bytes sent
   */
  static size_t ZMQSend(zmq::socket_t *sock, const void *data, size_t len,
                        int flag = 0);

  // 0 means cannot be sent, try again;
  // should not happen unless flag = ZMQ_DONTWAIT
  static size_t ZMQSend(zmq::socket_t *sock, int32_t zmq_id, const void *data,
                        size_t len, int flag = 0);

  // msg is nollified during the call
  static size_t ZMQSend(zmq::socket_t *sock, zmq::message_t &msg,
                        int flag = 0);

  static size_t ZMQSend(zmq::socket_t *sock, int32_t zmq_id,
                        zmq::message_t &msg, int flag = 0);
};

}  // namespace paddle
