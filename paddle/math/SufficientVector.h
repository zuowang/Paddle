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

#include "paddle/math/Matrix.h"
#include "ParameterService.pb.h"

namespace paddle {

class SufficientVector {
public:
  SufficientVector() : u_(nullptr), v_(nullptr) {}
  ~SufficientVector();

  void FromProto(const ProtoSV& proto);

  void ToProto(ProtoSV* proto) const;

  void SetU(const MatrixPtr& u);

  void SetV(const MatrixPtr& v);

  MatrixPtr& GetU() { return u_; }

  MatrixPtr& GetV() { return v_; }
private:
  // M = u_ x v_
  MatrixPtr u_;
  MatrixPtr v_;
};

}  // namespace paddle

