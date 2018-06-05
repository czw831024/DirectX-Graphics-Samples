//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************
#pragma once
#include "RaytracingHlslCompat.h"
#ifdef HLSL
#include "ShaderUtil.hlsli"
#endif

struct InputConstants
{
    uint NumberOfElements;
    uint PerformUpdate;
};

// UAVs
#define OutputBVHRegister 0
#define ScratchBufferRegister 1
#define ChildNodesProcessedBufferRegister 2
#define HierarchyBufferRegister 3
#define AABBParentBufferRegister 4

#define GlobalDescriptorHeapRegister 0
#define GlobalDescriptorHeapRegisterSpace 1

// CBVs
#define InputConstantsRegister 0

#ifdef HLSL
globallycoherent RWByteAddressBuffer outputBVH : UAV_REGISTER(OutputBVHRegister);
RWByteAddressBuffer scratchMemory : UAV_REGISTER(ScratchBufferRegister);
RWByteAddressBuffer childNodesProcessedCounter : UAV_REGISTER(ChildNodesProcessedBufferRegister);
RWStructuredBuffer<HierarchyNode> hierarchyBuffer : UAV_REGISTER(HierarchyBufferRegister);
RWStructuredBuffer<uint> aabbParentBuffer : UAV_REGISTER(AABBParentBufferRegister);
RWByteAddressBuffer DescriptorHeapBufferTable[] : UAV_REGISTER_SPACE(GlobalDescriptorHeapRegister, GlobalDescriptorHeapRegisterSpace);

cbuffer ConstructHierarchyConstants : CONSTANT_REGISTER(InputConstantsRegister)
{
    InputConstants Constants;
};
#endif
