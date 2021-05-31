// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/ie_parsed_network.hpp>

#include <string>

#include <legacy/details/ie_cnn_network_tools.h>
#include <caseless.hpp>

#include <vpu/compile_env.hpp>
#include <frontend.hpp>
namespace vpu {

IeParsedNetwork parseNetwork(const ie::CNNNetwork& network) {
    VPU_PROFILE(parseNetwork);

    const auto& env = CompileEnv::get();
    ie::details::CaselessEq<std::string> cmp;

    env.log->trace("Parse IE network : %s", network.getName());
    VPU_LOGGER_SECTION(env.log);

    IeParsedNetwork out;
    out.networkInputs = network.getInputsInfo();
    out.networkOutputs = network.getOutputsInfo();

    env.log->trace("Got %d inputs and %d outputs", out.networkInputs.size(), out.networkOutputs.size());
    IE_ASSERT(!out.networkInputs.empty());
    IE_ASSERT(!out.networkOutputs.empty());

    env.log->trace("Perform topological sort");
    const auto sortedNodes = network.getFunction()->get_ordered_ops();
    IE_ASSERT(!sortedNodes.empty());

    for (const auto& node : sortedNodes) {
        VPU_LOGGER_SECTION(env.log);

        IE_ASSERT(node != nullptr);

        if (cmp(node->get_type_name(), "Input")) {
            env.log->trace("Found Input layer : %s", node->get_friendly_name());
            continue;
        }

        if (cmp(node->get_type_name(), "Const")) {
            env.log->trace("Found Const layer : %s", node->get_friendly_name());

            if (node->get_output_size() != 1) {
                VPU_THROW_FORMAT(
                    "Const layer %v has unsupported number of outputs %v",
                    node->get_friendly_name(), node->get_output_size());
            }

            if (layer->blobs.size() != 1) {
                VPU_THROW_FORMAT(
                    "Const layer %v has unsupported number of blobs %v",
                    layer->name, layer->blobs.size());
            }

            const auto constData = layer->outData[0];
            IE_ASSERT(constData != nullptr);

            const auto constBlob = layer->blobs.begin()->second;
            IE_ASSERT(constBlob != nullptr);

            out.constDatas.emplace_back(constData, constBlob);

            continue;
        }

        env.log->trace("Found plain layer : %s", layer->name);
        out.orderedLayers.push_back(layer);
    }

    return out;
}

}  // namespace vpu
