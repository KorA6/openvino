// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/ie_parsed_network.hpp>

#include <string>

#include <legacy/details/ie_cnn_network_tools.h>
#include <caseless.hpp>

#include <vpu/compile_env.hpp>

namespace vpu {

IeParsedNetwork parseNetwork(const ie::CNNNetwork& network) {
    VPU_PROFILE(parseNetwork);

    const auto& env = CompileEnv::get();

    env.log->trace("Parse IE network : %s", network.getName());
    VPU_LOGGER_SECTION(env.log);

    IeParsedNetwork out;
    out.networkInputs = network.getInputsInfo();
    out.networkOutputs = network.getOutputsInfo();
    // out.networkParameters = network.getFunction()->get_parameters();
    // out.networkParameters = network.getFunction()->get_parameters();
    env.log->trace("Got %d inputs and %d outputs", out.networkInputs.size(), out.networkOutputs.size());
    IE_ASSERT(!out.networkInputs.empty());
    IE_ASSERT(!out.networkOutputs.empty());

    // looks unnecessary 
    env.log->trace("Perform topological sort");
    const auto sortedNodes = network.getFunction()->get_ordered_ops();
    IE_ASSERT(!sortedNodes.empty());
    for (const auto& node : sortedNodes) {
        VPU_LOGGER_SECTION(env.log);
        std::cout << "node :" << node->get_friendly_name() << std::endl;
        IE_ASSERT(node != nullptr);
        if (ngraph::as_type_ptr<ngraph::op::Parameter>(node)) {
            env.log->trace("Found Parameter node : %s", node->get_friendly_name());
            out.networkParameters.emplace_back(node);
            std::cout << node->get_friendly_name() << " " << out.networkParameters.size() << std::endl;
            continue;
        }
        if (ngraph::as_type_ptr<ngraph::op::Result>(node)) {
            env.log->trace("Found Result node : %s", node->get_friendly_name());
            out.networkResults.emplace_back(node);
            continue;
        }

        if (ngraph::as_type_ptr<ngraph::op::Constant>(node)) {
            env.log->trace("Found Const layer : %s", node->get_friendly_name());
            if (node->get_output_size() != 1) {
                VPU_THROW_FORMAT(
                    "Const layer %v has unsupported number of outputs %v",
                    node->get_friendly_name(), node->get_output_size());
            }

            // const auto constData = layer->outData[0];
            // IE_ASSERT(constData != nullptr);

            // const auto constBlob = shareWe;
            // IE_ASSERT(constBlob != nullptr);

            out.constDatas.emplace_back(node);

            continue;
        }

        env.log->trace("Found plain layer : %s", node->get_friendly_name());
        out.orderedOps.push_back(node);
    }

    return out;
}

}  // namespace vpu
