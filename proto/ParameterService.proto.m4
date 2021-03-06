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

import "ParameterConfig.proto";
import "TrainerConfig.proto";

package paddle;

/**
 * Various structs for communicating with parameter server
 */

enum ParameterUpdateMode {
  // Set parameter
   PSERVER_UPDATE_MODE_SET_PARAM = 0;//use local param
   PSERVER_UPDATE_MODE_SET_PARAM_ZERO = 1;//set zero param

  // Update parameter once a gradient is received
  PSERVER_UPDATE_MODE_ASYNC_SGD = 2;

  // Accumulate gradient
  PSERVER_UPDATE_MODE_ADD_GRADIENT = 3;

  // Average parameters
  PSERVER_UPDATE_MODE_AVERAGE_PARAMETER = 4;

  // No update. Only get parameters back.
  PSERVER_UPDATE_MODE_GET_PARAM = 5;
  PSERVER_UPDATE_MODE_GET_PARAM_SPARSE = 6;//only get sparse rows

  // Average gradient
  PSERVER_UPDATE_MODE_AVERAGE_GRADIENT = 7;
};

message ParameterBlock {
  // it accurately means parameter id.
  required uint64 para_id = 1;
  // global sparse row or dense block for each block in parameter
  required uint64 block_id = 2;
  // offset in (local) storage
  required uint64 begin_pos = 3;
  // actual size of block, size for last block is [endDim -beginDim],
  // others is parameter_block_size in ParameterConfig
  required uint64 block_size = 4;
}

enum PServerStatus {
  PSERVER_STATUS_NOT_SET = 0;
  PSERVER_STATUS_PARAMETER_READY = 1;
};

enum BatchStatus {
  BATCH_START = 0;
  BATCH_ON = 1;
  BATCH_FINISH = 2;
  BATCH_START_AND_FINISH = 3;
};

message SendParameterRequest {
  required ParameterUpdateMode update_mode = 1;
  repeated ParameterBlock blocks = 2;
  required bool send_back_parameter = 3;

  // number of samples used for calculating this update
  optional int64 num_samples = 4;

  // cost will be used to calculate global objective value
  optional real cost = 5;

  required BatchStatus batch_status = 6;

  optional int32 trainer_id = 7;

  // send back parameter type on pserver, PARAMETER_VALUE by default
  optional int32 send_back_parameter_type = 8 [default = 0];

  // forwardbackward time in usec
  optional uint64 forwardbackward_time = 9;

}

message WaitPassStartRequest {
}

message WaitPassStartResponse {
}

message WaitPassFinishRequest {
}

message WaitPassFinishResponse {
}

message WaitStageStartRequest {
}

message WaitStageStartResponse {
}

message WaitStageFinishRequest {
}

message WaitStageFinishResponse {
}

enum SyncObject {
  SYNC_DEFAULT = 0; // wait for the synchronizeBarrier_
  SYNC_DATA = 1; // wait for the synchronizeDataBarrier_
  SYNC_STAGE = 2;
}

message SynchronizeRequest {
  required SyncObject sync_object_id = 1 [default = SYNC_DEFAULT];

  optional int32 trainer_id = 2;
}

message SynchronizeResponse {
}

message SendParameterResponse  {
  repeated ParameterBlock blocks = 1;
}

message SetConfigRequest {
  repeated ParameterConfig param_configs = 1;
  required OptimizationConfig opt_config = 2;
  required string save_dir = 4;
  required int32 server_id = 5;
  required bool is_sparse_server = 6;
}

message SetConfigResponse{
}

message GetStatusRequest {
}

message GetStatusResponse {
  required PServerStatus status = 1;
}

message SetStatusRequest {
  required PServerStatus status = 1;
}

message SetStatusResponse {
}

// create a column vector. The size is the dimension of parameter
message CreateVectorRequest {
}

message CreateVectorResponse {
  // error message. Empty if success
  optional string return_message = 1;

  required int64 handle = 2;
}

message ReleaseVectorRequest {
  required int64 handle = 1;
}

message ReleaseVectorResponse {
  // error message. Empty if success
  optional string return_message = 1;
}

// Create a column major matrix. The number of rows is the dimension
// of parameter. The number of columns is specifed by num_cols
message CreateMatrixRequest {
  required int32 num_cols = 1;
}

message CreateMatrixResponse {
  // error message. Empty if success
  optional string return_message = 1;

  required int64 handle = 2;
}

message ReleaseMatrixRequest {
  required int64 handle = 1;
}

message ReleaseMatrixResponse {
  // error message. Empty if success
  optional string return_message = 1;
}


/**
 * The operations are defined using the variables commented at Operation
 * and OperationResult
 */
enum MatrixVectorOperation {
  // r = u^T u
  PSERVER_OP_utu = 0;

  // r = u^T v
  PSERVER_OP_utv = 1;

  // u = a u
  PSERVER_OP_au = 2;

  // v = a u + b v
  PSERVER_OP_au_bv = 3;

  // u = a A x + b u
  PSERVER_OP_aAx_bu = 4;

  // Stochastic gradient update
  PSERVER_OP_SGD = 5;

  // u = a
  PSERVER_OP_RESET = 6;

  // v = u
  PSERVER_OP_COPY = 7;

  // w = a u + b v + c w
  PSERVER_OP_au_bv_cw = 8;

  // owlqn: MakeSteepestDescDir
  PSERVER_OP_MAKE_STEEPEST_DESC_DIR = 9;

  // owlqn: FixDirSigns
  PSERVER_OP_FIX_DIR_SIGNS = 10;

  // owlqn: DirDeriv
  PSERVER_OP_DIR_DERIV = 11;

  // owlqn: FixOmegaSigns
  PSERVER_OP_FIX_OMEGA_SIGNS = 12;

  // Get overall cost
  PSERVER_OP_COST = 13;

  // Pass control
  PSERVER_OP_START_PASS = 14;
  PSERVER_OP_FINISH_PASS = 15;

  // randomize value
  PSERVER_OP_RANDOMIZE = 16;

  // call optimizer apply
  PSERVER_OP_APPLY = 17;
}

message ProtoVector {
  required int64 dim = 1;
  repeated real values = 2 [packed = true];
}

message ProtoMatrix {
  required int64 num_rows = 1;
  required int64 num_cols = 2;
  repeated real values = 3 [packed = true];
}

message Operation {
  required MatrixVectorOperation operation = 1;

  // vector handles created on the pserver
  repeated int64 pvectors = 2;        // u, v, w

  // matrix handles created on the pserver
  repeated int64 pmatrices = 3;       // A, B, C

  repeated real scalars = 4;  	      // a, b, c
  repeated ProtoVector vectors = 5;   // x, y, z
  repeated ProtoMatrix matrices = 6;  // X, Y, Z
}

message OperationResult {
  // error message. Empty if success
  optional string return_message = 1;
//
  repeated real scalars = 2;  // d, e, f
  repeated ProtoVector vectors = 3;  // p, q, r
  repeated ProtoMatrix matrices = 4;  // P, Q, R
}

message DoOperationRequest {
  repeated Operation operations = 1;

  // If true, wait for gradient to be ready before starting the operations
  required bool wait_for_gradient = 2;

  // If true, send back the parameter to clients after the operations are
  // finished
  required bool send_back_parameter = 3;

  // If true, and if all clients call waitPassFinish,
  // signal all clients finish the pass
  required bool release_pass = 4;
}

message DoOperationResponse {
  // error message. Empty if success
  optional string return_message = 1;

  repeated OperationResult results = 2;

  required bool pass_finish = 3;
}

message LoadValueRequest {
  required string dir_name = 1;
}

message LoadValueResponse {
  // error message. Empty if success
  optional string return_message = 1;
}

message SaveValueRequest {
  required string dir_name = 1;
}

message SaveValueResponse {
  // error message. Empty if success
  optional string return_message = 1;
}

enum DataUpdateMode {
  // Client send it's own data to pserver
  DATA_UPDATE_MODE_SET_OWN = 0;
  // Client get all user data from all pservers
  DATA_UPDATE_MODE_GET_ALL = 1;
  // Client send it's own ref feature to pserver
  DATA_UPDATE_MODE_SET_REF = 2;
  // Client get all ref featuers from all pservers
  DATA_UPDATE_MODE_GET_REF = 3;
  // Client send it's own ref label to pserver
  DATA_UPDATE_MODE_SET_REF_LABEL = 4;
  // Client get all ref labels from all pservers
  DATA_UPDATE_MODE_GET_REF_LABEL =5;
  // Client send it's own ref grad to pserver
  DATA_UPDATE_MODE_SET_REF_GRAD =6;
  // Client get all ref grad from all pservers
  DATA_UPDATE_MODE_GET_REF_GRAD =7;
}

enum SendDataType {
  DATA_REF = 0;
  DATA_REFLABEL = 1;
  DATA_REFGRAD = 2;
  DATA_REDUCE_SUM = 3;
}

enum TransDataType {
  TRANS_INT32 = 0;
  TRANS_UINT32_T = 1;
  TRANS_INT64_T = 2;
  TRANS_UINT64_T = 3;
  TRANS_FLOAT = 5;
  TRANS_DOUBLE = 6;
}

message DataBlock {
  // total byte size of this data blcok
  required uint64 total_size = 1;
  // byte size of one data type
  required int32 data_size = 2;
  // data_type
  optional TransDataType data_type = 3 [default = TRANS_DOUBLE];
}

message SendDataRequest {
  required SendDataType type = 1;
  required DataUpdateMode update_mode = 2;
  repeated DataBlock blocks = 3;
  required uint64 client_id = 4;
  required uint64 server_id = 5;
}

message SendDataResponse {
  required SendDataType type = 1;
  repeated DataBlock blocks = 2;
  required uint64 server_id = 3;
}
