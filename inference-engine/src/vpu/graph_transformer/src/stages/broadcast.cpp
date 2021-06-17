// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/utils/numeric.hpp>

#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpu {

namespace {

class BroadcastStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<BroadcastStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto inputOrder = input(0)->desc().dimsOrder();
        auto outputOrder = DimsOrder::fromNumDims(output(0)->desc().numDims());

        if (inputOrder.numDims() >= 3 && inputOrder.dimInd(Dim::C) == 0) {
            outputOrder.moveDim(Dim::C, 0);
        }

        orderInfo.setOutput(outputEdge(0), outputOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement().remove(0));
        stridesInfo.setOutput(outputEdge(0), StridesRequirement().remove(0));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NotNeeded;
    }

    void initialCheckImpl() const override {
        const auto mode = attrs().getOrDefault<BroadcastMode>("mode", BroadcastMode::NUMPY);
        const auto& dataPrecision = input(0)->desc().type();

        VPU_THROW_UNLESS(numOutputs() == 1,
                         "{} stage with name {} must have only 1 output, actually provided {} outputs",
                         type(), name(), numOutputs());
        if (mode == BroadcastMode::EXPLICIT) {
            VPU_THROW_UNLESS(numInputs() == 3,
                             "{} stage with name {} and explicit mode must have 3 inputs, actually "
                             "provided {} inputs", type(), name(), numInputs());
            assertInputsOutputsTypes(this,
                                     {{dataPrecision}, {DataType::S32}, {DataType::S32}},
                                     {{dataPrecision}});
        } else {
            VPU_THROW_UNLESS(numInputs() == 2,
                             "{} stage with name {} and numpy or bidirectional mode must have 2 inputs, actually "
                             "provided {} inputs", type(), name(), numInputs());
            assertInputsOutputsTypes(this,
                                     {{dataPrecision}, {DataType::S32}},
                                     {{dataPrecision}});
        }
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto mode = attrs().getOrDefault<BroadcastMode>("mode", BroadcastMode::NUMPY);
        serializer.append(mode);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        const auto mode = attrs().getOrDefault<BroadcastMode>("mode", BroadcastMode::NUMPY);

        input(0)->serializeBuffer(serializer);
        input(1)->serializeBuffer(serializer);
        if (mode == BroadcastMode::EXPLICIT) {
            input(2)->serializeBuffer(serializer);
        }
        output(0)->serializeBuffer(serializer);
    }
};

}  // namespace
static std::string getModeAsString(const ngraph::op::BroadcastType mode) {

    switch (mode)
    {
    case ngraph::op::BroadcastType::EXPLICIT :
        return "explicit";
    case ngraph::op::BroadcastType::BIDIRECTIONAL :
        return "bidirectional";
    case ngraph::op::BroadcastType::PDPD :
        return "pdpd";
    case ngraph::op::BroadcastType::NUMPY :
        return "numpy";
    default:
        return "";
    }
}
void FrontEnd::parseBroadcast(
        const Model& model,
        const NodePtr& node,
        const DataVector& inputs,
        const DataVector& outputs) const {
    auto broadcast = ngraph::as_type_ptr<ngraph::opset4::Broadcast>(node);
    VPU_THROW_UNLESS(broadcast != nullptr,
                     "parseBroadcast expects valid NodePtr, got nullptr");
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "{} layer with name {} must have only 1 output, actually provided {} outputs",
                     broadcast->get_type_name(), broadcast->get_name(), outputs.size());
    const auto output = outputs[0];
    const auto broadcastMode =  broadcast->get_broadcast_spec().m_type;
    std::string modeString = getModeAsString(broadcastMode);
    const std::map<std::string, BroadcastMode> modeFromString = {
        {"numpy", BroadcastMode::NUMPY},
        {"explicit", BroadcastMode::EXPLICIT},
        {"bidirectional", BroadcastMode::BIDIRECTIONAL}
    };
    const auto& modeFind = modeFromString.find(modeString);
    VPU_THROW_UNLESS(modeFind != modeFromString.end(),
                     "{} layer with name {}: Graph Transformer doesn't support {} mode",
                     node->get_type_name(), node->get_friendly_name(), modeString);
    const auto mode = modeFind->second;
    if (mode == BroadcastMode::NUMPY || mode == BroadcastMode::BIDIRECTIONAL) {
        VPU_THROW_UNLESS(inputs.size() == 2,
                         "{} layer with name {} and {} mode must have 2 inputs, actually "
                         "provided {} inputs", node->get_type_name(), node->get_friendly_name(), modeString, inputs.size());
    } else if (mode == BroadcastMode::EXPLICIT) {
        VPU_THROW_UNLESS(inputs.size() == 3,
                         "{} layer with name {} and explicit mode must have 3 inputs, actually "
                         "provided {} inputs", node->get_type_name(), node->get_friendly_name(), inputs.size());
        const auto axesMappingDesc = inputs[2]->desc();
        const auto axesMappingPerm = axesMappingDesc.dimsOrder().toPermutation();
        const auto axesMappingDim = axesMappingDesc.dim(axesMappingPerm.at(0));
        VPU_THROW_UNLESS(axesMappingDesc.numDims() == 1,
                         "{} layer with name {} and explicit mode must have 1D axesMapping tensor, "
                         "actually provided {}D tensor",
                         node->get_type_name(), node->get_friendly_name(), axesMappingDesc.numDims());
        VPU_THROW_UNLESS(axesMappingDim == inputs[0]->desc().numDims(),
                         "{} layer with name {} and explicit mode must have axesMapping tensor with "
                         "size equals to number of output dims, expected [{}], provided [{}]",
                         node->get_type_name(), node->get_friendly_name(), output->desc().numDims(), axesMappingDim);

    } else {
        VPU_THROW_FORMAT("{} layer with name {}: Graph Transformer doesn't support {} mode",
                         node->get_type_name(), node->get_friendly_name(), modeString);
    }

    const auto shape = inputs[1];
    const auto shapeDesc = inputs[1]->desc();
    const auto shapeDim = shapeDesc.dim(shapeDesc.dimsOrder().toPermutation().at(0));
    VPU_THROW_UNLESS(shapeDesc.numDims() == 1,
                     "{} layer with name {} and explicit mode must have 1D target shape tensor, "
                     "actually provided {}D tensor",
                     node->get_type_name(), node->get_friendly_name(), shapeDesc.numDims());
    VPU_THROW_UNLESS(shapeDim == output->desc().numDims() || mode != BroadcastMode::EXPLICIT,
                     "{} layer with name {} and explicit mode must have target shape tensor with "
                     "size equals to number of output dims, expected [{}], provided [{}]",
                     node->get_type_name(), node->get_friendly_name(), output->desc().numDims(), shapeDim);

    auto stage = model->addNewStage<BroadcastStage>(
            node->get_friendly_name(),
            StageType::Broadcast,
            node,
            inputs,
            outputs);

    stage->attrs().set("mode", mode);
}

}  //namespace vpu
