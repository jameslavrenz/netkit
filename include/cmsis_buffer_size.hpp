#pragma once

#include <cstddef>
#include <cstdint>

std::size_t CmsisConv2dWorkspaceBytes(uint32_t in_h,
                                      uint32_t in_w,
                                      int kernel_size,
                                      int stride,
                                      int pad_h,
                                      int pad_w,
                                      int in_channels,
                                      int out_channels);

std::size_t CmsisDepthwiseConv2dWorkspaceBytes(uint32_t in_h,
                                               uint32_t in_w,
                                               int kernel_h,
                                               int kernel_w,
                                               int stride,
                                               int pad_h,
                                               int pad_w,
                                               int channels);

std::size_t CmsisGeluWorkspaceBytes(uint32_t num_elements);

void CmsisBeginKernelWorkspacePlan(std::size_t* max_out);
void CmsisEndKernelWorkspacePlan();

void CmsisBumpConv2dWorkspace(uint32_t in_h,
                              uint32_t in_w,
                              int kernel_size,
                              int stride,
                              int pad_h,
                              int pad_w,
                              int in_channels,
                              int out_channels);

void CmsisBumpDepthwiseConv2dWorkspace(uint32_t in_h,
                                       uint32_t in_w,
                                       int kernel_h,
                                       int kernel_w,
                                       int stride,
                                       int pad_h,
                                       int pad_w,
                                       int channels);

void CmsisBumpGeluWorkspace(uint32_t num_elements);
