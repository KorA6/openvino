// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_elements_update_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "scatter_update/scatter_elements_update_kernel_selector.h"
#include "scatter_update/scatter_elements_update_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace gpu {
kernel_selector::scatter_update_axis convert_axis(scatter_elements_update::scatter_elements_update_axis axis, const scatter_elements_update_node& arg) {
    switch (axis) {
        case scatter_elements_update::along_x:
            return kernel_selector::scatter_update_axis::X;
        case scatter_elements_update::along_y:
            return kernel_selector::scatter_update_axis::Y;
        case scatter_elements_update::along_z:
            return kernel_selector::scatter_update_axis::Z;
        case scatter_elements_update::along_w:
            return kernel_selector::scatter_update_axis::W;
        case scatter_elements_update::along_f:
            return kernel_selector::scatter_update_axis::FEATURE;
        case scatter_elements_update::along_b:
            return kernel_selector::scatter_update_axis::BATCH;
        default:
            CLDNN_ERROR_MESSAGE(arg.id(), "Unsupported Axis");
    }
    return kernel_selector::scatter_update_axis::X;
}

struct scatter_elements_update_gpu : typed_primitive_gpu_impl<scatter_elements_update> {
    using parent = typed_primitive_gpu_impl<scatter_elements_update>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scatter_elements_update_gpu>(*this);
    }

public:
    static primitive_impl* create(const scatter_elements_update_node& arg) {
        auto scatter_elements_update_params = get_default_params<kernel_selector::scatter_elements_update_params>(arg);
        auto scatter_elements_update_optional_params =
            get_default_optional_params<kernel_selector::scatter_elements_update_optional_params>(arg.get_program());

        scatter_elements_update_params.axis = convert_axis(arg.get_primitive()->axis, arg);

        scatter_elements_update_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));
        scatter_elements_update_params.inputs.push_back(convert_data_tensor(arg.input(2).get_output_layout()));

        auto& kernel_selector = kernel_selector::scatter_elements_update_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(scatter_elements_update_params, scatter_elements_update_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto scatter_elements_update = new scatter_elements_update_gpu(arg, best_kernels[0]);

        return scatter_elements_update;
    }
};

namespace detail {

attach_scatter_elements_update_gpu::attach_scatter_elements_update_gpu() {
    auto val_fw = scatter_elements_update_gpu::create;
    implementation_map<scatter_elements_update>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<scatter_elements_update>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<scatter_elements_update>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);

    implementation_map<scatter_elements_update>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<scatter_elements_update>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<scatter_elements_update>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);

    implementation_map<scatter_elements_update>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<scatter_elements_update>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    implementation_map<scatter_elements_update>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
