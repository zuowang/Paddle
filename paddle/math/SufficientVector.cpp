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



#include "SufficientVector.h"

namespace paddle {

template <class T>
void copyToRepeatedField(google::protobuf::RepeatedField<T>* dest, const T* src,
                         size_t size) {
  dest->Clear();
  dest->Reserve(size);

  for (size_t i = 0; i < size; ++i) {
    dest->AddAlreadyReserved(src[i]);
  }
}

SufficientVector::~SufficientVector() {
  u_.reset();
  v_.reset();
}

void SufficientVector::FromProto(const ProtoSV& proto) {
  const ProtoMatrix& pmatU = proto.u();
  u_ = Matrix::create(
      pmatU.num_rows(), pmatU.num_cols(), false, false);
  real* data = u_->getData();
  for (int i = 0; i < pmatU.num_cols() * pmatU.num_rows(); ++i) {
    data[i] = pmatU.values(i);
  }
  const ProtoMatrix& pmatV = proto.v();
  v_ = Matrix::create(
      pmatV.num_rows(), pmatV.num_cols(), false, false);
  real* dataV = v_->getData();
  for (int i = 0; i < pmatV.num_cols() * pmatV.num_rows(); ++i) {
    dataV[i] = pmatV.values(i);
  }
}

void SufficientVector::ToProto(ProtoSV* proto) const {
  ProtoMatrix& pmatU = *proto->mutable_u();
  pmatU.set_num_cols(u_->getWidth());
  pmatU.set_num_rows(u_->getHeight());
  copyToRepeatedField(pmatU.mutable_values(), u_->getData(),
                      pmatU.num_cols() * pmatU.num_rows());

  ProtoMatrix& pmatV = *proto->mutable_v();
  pmatV.set_num_cols(v_->getWidth());
  pmatV.set_num_rows(v_->getHeight());
  copyToRepeatedField(pmatV.mutable_values(), v_->getData(),
                      pmatV.num_cols() * pmatV.num_rows());
}

void SufficientVector::SetU(const MatrixPtr& u) {
  u_ = Matrix::create(
      u->getHeight(), u->getWidth(), false, false);
  real* data = u_->getData();
  for (size_t i = 0; i < u->getWidth() * u->getHeight(); ++i) {
    data[i] = u->getData()[i];
  }
}

void SufficientVector::SetV(const MatrixPtr& v) {
  v_ = Matrix::create(
      v->getHeight(), v->getWidth(), false, false);
  real* data = v_->getData();
  for (size_t i = 0; i < v->getWidth() * v->getHeight(); ++i) {
    data[i] = v->getData()[i];
  }
}

}  // namespace paddle

