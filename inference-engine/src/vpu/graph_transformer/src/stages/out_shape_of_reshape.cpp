// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/model/data_contents/ie_blob_content.hpp>
#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"
#include <vector>
#include <map>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class OutShapeOfReshapeStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<OutShapeOfReshapeStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
                                 {{DataType::S32}, {DataType::S32}},
                                 {{DataType::S32}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto specialZero = attrs().get<bool>("specialZero");

        serializer.append(static_cast<int32_t>(specialZero));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        input(0)->serializeBuffer(serializer);
        input(1)->serializeBuffer(serializer);
        output(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseOutShapeOfReshape(
        const Model& model,
        const NodePtr& node,
        const DataVector& inputs,
        const DataVector& outputs) const {
    auto outShapeOfReshape = ngraph::as_type_ptr<ngraph::vpu::op::OutShapeOfReshape>(node);
    IE_ASSERT(outShapeOfReshape != nullptr);
    VPU_THROW_UNLESS(inputs.size() == 2,
                     "OutShapeOfReshape stage with name %s must have only 2 inputs, "
                     "actually provided %d", outShapeOfReshape->get_name(), inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "OutShapeOfReshape stage with name %s must have only 1 output, "
                     "actually provided %d", outShapeOfReshape->get_name(), outputs.size());

    auto inDataShape = inputs[0];
    auto outShapeDescriptor = inputs[1];
    auto outDataShape = outputs[0];

    VPU_THROW_UNLESS(inDataShape->desc().numDims() == 1,
                     "OutShapeOfReshape stage with name %s must have 1D input data shape tensor, "
                     "actually provided %dD tensor", outShapeOfReshape->get_name(), inDataShape->desc().numDims());
    VPU_THROW_UNLESS(outShapeDescriptor->desc().numDims() == 1,
                     "OutShapeOfReshape stage with name %s must have 1D output shape descriptor "
                     "tensor, actually provided %dD tensor",
                     outShapeOfReshape->get_name(), outShapeDescriptor->desc().numDims());
    VPU_THROW_UNLESS(outDataShape->desc().numDims() == 1,
                     "OutShapeOfReshape stage with name %s must have 1D output data shape tensor, "
                     "actually provided %dD tensor", outShapeOfReshape->get_name(), outDataShape->desc().numDims());

    VPU_THROW_UNLESS(outShapeDescriptor->desc().totalDimSize() == outDataShape->desc().totalDimSize(),
                     "OutShapeOfReshape stage with name %s must have output shape descriptor and "
                     "output data shape tensor with equal length, actually provided %d vs %d",
                     outShapeOfReshape->get_name(), outShapeDescriptor->desc().totalDimSize(),
                     outDataShape->desc().totalDimSize());


    auto outShapeOfReshapeStage = model->addNewStage<OutShapeOfReshapeStage>(
            outShapeOfReshape->get_name(),
            StageType::OutShapeOfReshape,
            outShapeOfReshape,
            inputs,
            outputs);

    auto specialZero = outShapeOfReshape->getSpecialZero();
    outShapeOfReshapeStage->attrs().set<bool>("specialZero", specialZero);
}

}  // namespace vpu
