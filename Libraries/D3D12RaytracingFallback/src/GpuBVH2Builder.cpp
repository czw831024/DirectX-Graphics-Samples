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
#include "pch.h"

#pragma once

namespace FallbackLayer
{
    GpuBvh2Builder::GpuBvh2Builder(ID3D12Device *pDevice, UINT totalLaneCount, UINT nodeMask) :
        m_sceneAABBCalculator(pDevice, nodeMask),
        m_mortonCodeCalculator(pDevice, nodeMask),
        m_sorterPass(pDevice, nodeMask),
        m_rearrangePass(pDevice, nodeMask),
        m_loadInstancesPass(pDevice, nodeMask),
        m_loadPrimitivesPass(pDevice, nodeMask),
        m_constructHierarchyPass(pDevice, nodeMask),
        m_constructAABBPass(pDevice, nodeMask),
        m_postBuildInfoQuery(pDevice, nodeMask),
        m_copyPass(pDevice, totalLaneCount, nodeMask),
        m_treeletReorder(pDevice, nodeMask)
    {}

    void GpuBvh2Builder::BuildRaytracingAccelerationStructure(
        _In_  ID3D12GraphicsCommandList *pCommandList,
        _In_  const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC *pDesc,
        _In_ ID3D12DescriptorHeap *pCbvSrvUavDescriptorHeap)
    {
#ifdef DEBUG
        D3D12_GET_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO_DESC prebuildInfoDesc = {};
        prebuildInfoDesc.DescsLayout = pDesc->DescsLayout;
        prebuildInfoDesc.Flags = pDesc->Flags;
        prebuildInfoDesc.NumDescs = pDesc->NumDescs;
        prebuildInfoDesc.pGeometryDescs = pDesc->pGeometryDescs;
        prebuildInfoDesc.ppGeometryDescs = pDesc->ppGeometryDescs;
        prebuildInfoDesc.Type = pDesc->Type;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildOutput;

        CComPtr<ID3D12Device> pDevice;
        pCommandList->GetDevice(IID_PPV_ARGS(&pDevice));

        GetRaytracingAccelerationStructurePrebuildInfo(&prebuildInfoDesc, &prebuildOutput);
        if (pDesc->DestAccelerationStructureData.SizeInBytes < prebuildOutput.ResultDataMaxSizeInBytes)
        {
            ThrowFailure(E_INVALIDARG, L"DestAccelerationStructureData.SizeInBytes too small, "
                L"ensure the size matches up with a size returned from "
                L"EmitRaytracingAccelerationStructurePostBuildInfo/GetRaytracingAccelerationStructurePrebuildInfo");
        }

        if (pDesc->ScratchAccelerationStructureData.SizeInBytes < prebuildOutput.ScratchDataSizeInBytes)
        {
            ThrowFailure(E_INVALIDARG, L"pDesc->ScratchAccelerationStructureData.SizeInBytes too small, "
                L"ensure the size matches up with a size returned from "
                L"EmitRaytracingAccelerationStructurePostBuildInfo/GetRaytracingAccelerationStructurePrebuildInfo");
        }
#endif

        switch (pDesc->Type)
        {
            case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL:
                BuildBottomLevelBVH(pCommandList, pDesc);
            break;
            case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL:
                BuildTopLevelBVH(pCommandList, pDesc, pCbvSrvUavDescriptorHeap);
            break;
            default:
                ThrowFailure(E_INVALIDARG, L"Unrecognized D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE provided");
        }
    }

    void GpuBvh2Builder::LoadBVHGPUStreet(
        _In_  const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC *pDesc,
        Level bvhLevel,
        UINT numElements,
        BVHGPUStreet &street)
    {
        D3D12_GPU_VIRTUAL_ADDRESS bvhGpuVA = pDesc->DestAccelerationStructureData.StartAddress;
        ScratchMemoryPartitions scratchMemoryPartition = CalculateScratchMemoryUsage(bvhLevel, numElements);
        D3D12_GPU_VIRTUAL_ADDRESS scratchGpuVA = pDesc->ScratchAccelerationStructureData.StartAddress;
        
        street.scratchElementBuffer = scratchGpuVA + scratchMemoryPartition.OffsetToElements;
        street.mortonCodeBuffer = scratchGpuVA + scratchMemoryPartition.OffsetToMortonCodes;
        street.sceneAABB = scratchGpuVA + scratchMemoryPartition.OffsetToSceneAABB;
        street.sceneAABBScratchMemory = scratchGpuVA + scratchMemoryPartition.OffsetToSceneAABBScratchMemory;
        street.indexBuffer = scratchGpuVA + scratchMemoryPartition.OffsetToIndexBuffer;
        street.hierarchyBuffer = scratchGpuVA + scratchMemoryPartition.OffsetToHierarchy;
        street.calculateAABBScratchBuffer = scratchGpuVA + scratchMemoryPartition.OffsetToCalculateAABBDispatchArgs;
        street.nodeCountBuffer = scratchGpuVA + scratchMemoryPartition.OffsetToPerNodeCounter;

        switch(bvhLevel) 
        {
            case Level::Top:
            {
                UINT offsetFromElementsToMetadata = GetOffsetFromLeafNodesToBottomLevelMetadata(numElements);
                street.scratchMetadataBuffer = street.scratchElementBuffer + offsetFromElementsToMetadata;
                street.outputElementBuffer = bvhGpuVA + GetOffsetToLeafNodeAABBs(numElements);
                street.outputMetadataBuffer = street.outputElementBuffer + offsetFromElementsToMetadata;
                street.outputSortCacheBuffer = bvhGpuVA + GetOffsetToBVHSortedIndices(numElements);
                street.outputAABBParentBuffer = street.outputSortCacheBuffer + GetOffsetFromSortedIndicesToAABBParents(numElements);
            }
            break;
            case Level::Bottom:
            {
                street.scratchMetadataBuffer = street.scratchElementBuffer + GetOffsetFromPrimitivesToPrimitiveMetaData(numElements);
                street.outputElementBuffer = bvhGpuVA + GetOffsetToPrimitives(numElements);
                street.outputMetadataBuffer = street.outputElementBuffer + GetOffsetFromPrimitivesToPrimitiveMetaData(numElements);
                street.outputSortCacheBuffer = street.outputMetadataBuffer + GetOffsetFromPrimitiveMetaDataToSortedIndices(numElements);
                street.outputAABBParentBuffer = street.outputSortCacheBuffer + GetOffsetFromSortedIndicesToAABBParents(numElements);
            }
            break;
        }
    }

    void GpuBvh2Builder::BuildTopLevelBVH(
        _In_  ID3D12GraphicsCommandList *pCommandList,
        _In_  const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC *pDesc,
        _In_ ID3D12DescriptorHeap *pCbvSrvUavDescriptorHeap)
    {
        const SceneType sceneType = SceneType::BottomLevelBVHs;
        UINT numElements = pDesc->NumDescs;
        D3D12_GPU_DESCRIPTOR_HANDLE globalDescriptorHeap = pCbvSrvUavDescriptorHeap->GetGPUDescriptorHandleForHeapStart();

        BuildBVH(
            pCommandList,
            pDesc,
            Level::Top,
            sceneType,
            numElements,
            globalDescriptorHeap
        );
    }

    void GpuBvh2Builder::BuildBottomLevelBVH(
        _In_  ID3D12GraphicsCommandList *pCommandList,
        _In_  const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC *pDesc)
    {
        const SceneType sceneType = SceneType::Triangles;
        UINT numElements = GetTotalPrimitiveCount(*pDesc);
        D3D12_GPU_DESCRIPTOR_HANDLE globalDescriptorHeap = D3D12_GPU_DESCRIPTOR_HANDLE();

        BuildBVH(
            pCommandList,
            pDesc,
            Level::Bottom,
            sceneType,
            numElements,
            globalDescriptorHeap
        );
    }

    void GpuBvh2Builder::BuildBVH(
        _In_  ID3D12GraphicsCommandList *pCommandList,
        _In_  const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC *pDesc,
        Level bvhLevel,
        SceneType sceneType,
        UINT numElements,
        D3D12_GPU_DESCRIPTOR_HANDLE globalDescriptorHeap)
    {
        if (pDesc->DestAccelerationStructureData.StartAddress == 0)
        {
            ThrowFailure(E_INVALIDARG, L"DestAccelerationStructureData.StartAddress must be non-zero");
        }

        BVHGPUStreet street = {}; LoadBVHGPUStreet(pDesc, bvhLevel, numElements, street);

        const bool performUpdate = m_updateAllowed && (pDesc->Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE);

        // Load in the leaf-node elements of the BVH and calculate the entire scene's AABB.
        LoadBVHElements(
            pCommandList,
            pDesc,
            sceneType,
            numElements,
            performUpdate ? street.outputElementBuffer   : street.scratchElementBuffer, // If we're updating, write straight to output.
            performUpdate ? street.outputMetadataBuffer  : street.scratchMetadataBuffer, 
            performUpdate ? street.outputSortCacheBuffer : 0,
            street.sceneAABBScratchMemory,
            street.sceneAABB,
            globalDescriptorHeap);

        // If we don't have PERFORM_UPDATE set, rebuild the entire hierarchy.
        // (i.e. calc morton codes, sort, rearrange, build hierarchy, treelet reorder)
        if (!performUpdate) {
            BuildBVHHierarchy(
                pCommandList,
                pDesc,
                sceneType,
                numElements,
                street.scratchElementBuffer,
                street.outputElementBuffer,
                street.scratchMetadataBuffer,
                street.outputMetadataBuffer,
                street.sceneAABBScratchMemory,
                street.sceneAABB,
                street.mortonCodeBuffer,
                street.indexBuffer,
                m_updateAllowed ? street.outputSortCacheBuffer : 0,
                street.hierarchyBuffer,
                m_updateAllowed ? street.outputAABBParentBuffer : 0,
                street.nodeCountBuffer,
                globalDescriptorHeap);
        }

        // Fit AABBs around each node in the hierarchy.
        m_constructAABBPass.ConstructAABB(
            pCommandList,
            sceneType,
            pDesc->DestAccelerationStructureData.StartAddress,
            street.calculateAABBScratchBuffer,
            street.nodeCountBuffer,
            street.hierarchyBuffer,
            performUpdate ? street.outputAABBParentBuffer : 0,
            globalDescriptorHeap,
            numElements);
    }

    void GpuBvh2Builder::LoadBVHElements(
        _In_ ID3D12GraphicsCommandList *pCommandList,
        _In_  const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC *pDesc,
        const SceneType sceneType,
        const UINT numElements,
        D3D12_GPU_VIRTUAL_ADDRESS elementBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS metadataBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS indexBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS sceneAABBScratchMemory,
        D3D12_GPU_VIRTUAL_ADDRESS sceneAABB,
        D3D12_GPU_DESCRIPTOR_HANDLE globalDescriptorHeap)
    {
        switch(sceneType) 
        {
            case SceneType::BottomLevelBVHs:
            // Note that the load instances pass does load metadata even though it doesn't take a metadata
            // buffer address. Users don't specify BVH instance metadata, so the shader takes care of
            // putting the metadata where it needs to go on its own.
            m_loadInstancesPass.LoadInstances(
                pCommandList, 
                elementBuffer, 
                pDesc->InstanceDescs, 
                pDesc->DescsLayout, 
                numElements, 
                globalDescriptorHeap,
                indexBuffer);
            break;
            case SceneType::Triangles:
            // Load all the triangles into the bottom-level acceleration structure. This loading is done 
            // one VB/IB pair at a time since each VB will have unique characteristics (topology type/IB format)
            // and will generally have enough verticies to go completely wide
            m_loadPrimitivesPass.LoadPrimitives(
                pCommandList, 
                *pDesc, 
                numElements, 
                elementBuffer,
                metadataBuffer,
                indexBuffer);
            break;
        }

        m_sceneAABBCalculator.CalculateSceneAABB(
            pCommandList, 
            sceneType, 
            elementBuffer, 
            numElements, 
            sceneAABBScratchMemory, 
            sceneAABB);
    }

    void GpuBvh2Builder::BuildBVHHierarchy(
        _In_ ID3D12GraphicsCommandList *pCommandList,
        _In_  const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC *pDesc,
        const SceneType sceneType,
        const uint numElements,
        D3D12_GPU_VIRTUAL_ADDRESS scratchElementBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS outputElementBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS scratchMetadataBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS outputMetadataBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS sceneAABBScratchMemory,
        D3D12_GPU_VIRTUAL_ADDRESS sceneAABB,
        D3D12_GPU_VIRTUAL_ADDRESS mortonCodeBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS indexBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS outputSortCacheBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS hierarchyBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS outputAABBParentBuffer,
        D3D12_GPU_VIRTUAL_ADDRESS nodeCountBuffer,
        D3D12_GPU_DESCRIPTOR_HANDLE globalDescriptorHeap) 
    {
        m_mortonCodeCalculator.CalculateMortonCodes(
            pCommandList, 
            sceneType, 
            scratchElementBuffer, 
            numElements, 
            sceneAABB, 
            indexBuffer, 
            mortonCodeBuffer);

        m_sorterPass.Sort(
            pCommandList, 
            mortonCodeBuffer, 
            indexBuffer, 
            numElements, 
            false, 
            true);

        m_rearrangePass.Rearrange(
            pCommandList,
            sceneType,
            numElements,
            scratchElementBuffer,
            scratchMetadataBuffer,
            indexBuffer,
            outputElementBuffer,
            outputMetadataBuffer,
            outputSortCacheBuffer);

        m_constructHierarchyPass.ConstructHierarchy(
            pCommandList,
            sceneType,
            mortonCodeBuffer,
            hierarchyBuffer,
            outputAABBParentBuffer, // Store parent indices in hierarchy pass since AABBNodes don't store parent indices.
            globalDescriptorHeap,
            numElements);

        if (sceneType == SceneType::Triangles) 
        {
            m_treeletReorder.Optimize(
                pCommandList,
                numElements,
                hierarchyBuffer,
                outputAABBParentBuffer, // Make sure parent indices get updated when things are reshuffled
                nodeCountBuffer,
                sceneAABBScratchMemory,
                outputElementBuffer,
                globalDescriptorHeap,
                pDesc->Flags);
        }
    }

    void GpuBvh2Builder::CopyRaytracingAccelerationStructure(
        _In_  ID3D12GraphicsCommandList *pCommandList,
        _In_  D3D12_GPU_VIRTUAL_ADDRESS_RANGE DestAccelerationStructureData,
        _In_  D3D12_GPU_VIRTUAL_ADDRESS SourceAccelerationStructureData,
        _In_  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE Flags)
    {
        if (Flags == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE ||
            Flags == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT)
        {
            m_copyPass.CopyRaytracingAccelerationStructure(pCommandList, DestAccelerationStructureData, SourceAccelerationStructureData);
        }
        else
        {
            ThrowFailure(E_INVALIDARG,
                L"The only flags supported for CopyRaytracingAccelerationStructure are: "
                L"D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE/D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT");
        }
    }

    GpuBvh2Builder::ScratchMemoryPartitions GpuBvh2Builder::CalculateScratchMemoryUsage(Level level, UINT numPrimitives)
    {
#define ALIGN(alignment, num) (((num + alignment - 1) / alignment) * alignment)
#define ALIGN_GPU_VA_OFFSET(num) ALIGN(4, num)

        ScratchMemoryPartitions scratchMemoryPartitions = {};
        UINT64 &totalSize = scratchMemoryPartitions.TotalSize;
        UINT numInternalNodes = GetNumberOfInternalNodes(numPrimitives);
        UINT totalNumNodes = numPrimitives + numInternalNodes;

        scratchMemoryPartitions.OffsetToSceneAABB = totalSize;
        totalSize += ALIGN_GPU_VA_OFFSET(sizeof(AABB));

        const UINT sizePerElement = level == Level::Bottom ?
            sizeof(Primitive) + sizeof(PrimitiveMetaData) :
            (sizeof(AABBNode) + sizeof(BVHMetadata));
        scratchMemoryPartitions.OffsetToElements = totalSize;
        totalSize += ALIGN_GPU_VA_OFFSET(sizePerElement * numPrimitives);

        const UINT mortonCodeBufferSize = ALIGN_GPU_VA_OFFSET(sizeof(UINT) * numPrimitives);
        scratchMemoryPartitions.OffsetToMortonCodes = totalSize;

        const UINT indexBufferSize = ALIGN_GPU_VA_OFFSET(sizeof(UINT) * numPrimitives);
        scratchMemoryPartitions.OffsetToIndexBuffer = scratchMemoryPartitions.OffsetToMortonCodes + indexBufferSize;

        {
            // The scratch buffer used for calculating AABBs can alias over the MortonCode/IndexBuffer
            // because it's calculated before the MortonCode/IndexBuffer are needed. Additionally,
            // the AABB buffer used for treelet reordering is done after both stages so it can also alias
            scratchMemoryPartitions.OffsetToSceneAABBScratchMemory = scratchMemoryPartitions.OffsetToMortonCodes;
            INT64 sizeNeededToCalculateAABB = m_sceneAABBCalculator.ScratchBufferSizeNeeded(numPrimitives);
            INT64 sizeNeededForTreeletAABBs = TreeletReorder::RequiredSizeForAABBBuffer(numPrimitives);
            INT64 sizeNeededByMortonCodeAndIndexBuffer = mortonCodeBufferSize + indexBufferSize;
            UINT64 extraBufferSize = std::max(sizeNeededToCalculateAABB, std::max(sizeNeededForTreeletAABBs, sizeNeededByMortonCodeAndIndexBuffer));

            totalSize += extraBufferSize;
        }


        {
            UINT64 sizeNeededForAABBCalculation = 0;
            scratchMemoryPartitions.OffsetToCalculateAABBDispatchArgs = sizeNeededForAABBCalculation;
            sizeNeededForAABBCalculation += ALIGN_GPU_VA_OFFSET(sizeof(UINT) * numPrimitives);

            scratchMemoryPartitions.OffsetToPerNodeCounter = sizeNeededForAABBCalculation;
            sizeNeededForAABBCalculation += ALIGN_GPU_VA_OFFSET(sizeof(UINT) * (numInternalNodes));

            totalSize = std::max(sizeNeededForAABBCalculation, totalSize);
        }

        const UINT64 hierarchySize = ALIGN_GPU_VA_OFFSET(sizeof(HierarchyNode) * totalNumNodes);
        scratchMemoryPartitions.OffsetToHierarchy = totalSize;
        totalSize += hierarchySize;

        return scratchMemoryPartitions;
    }

    void GpuBvh2Builder::GetRaytracingAccelerationStructurePrebuildInfo(
        _In_  D3D12_GET_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO_DESC *pDesc,
        _Out_  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO *pInfo)
    {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE Type = pDesc->Type;
        UINT NumElements = pDesc->NumDescs;

        switch (Type)
        {
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL:
        {
            UINT totalNumberOfTriangles = GetTotalPrimitiveCount(*pDesc);
            const UINT numLeaves = totalNumberOfTriangles;
            // A full binary tree with N leaves will always have N - 1 internal nodes
            const UINT numInternalNodes = GetNumberOfInternalNodes(numLeaves);
            const UINT totalNumNodes = numLeaves + numInternalNodes;

            pInfo->ResultDataMaxSizeInBytes = sizeof(BVHOffsets) + totalNumberOfTriangles * (sizeof(Primitive) + sizeof(PrimitiveMetaData)) +
                totalNumNodes * sizeof(AABBNode);

            m_updateAllowed = (pDesc->Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) != 0;
            if (m_updateAllowed) 
            {
                pInfo->ResultDataMaxSizeInBytes += totalNumberOfTriangles * sizeof(UINT); // Saved sorted index buffer
                pInfo->ResultDataMaxSizeInBytes += totalNumNodes * sizeof(UINT); // Parent indices for AABBNodes
            }

            pInfo->ScratchDataSizeInBytes = CalculateScratchMemoryUsage(Level::Bottom, totalNumberOfTriangles).TotalSize;
            pInfo->UpdateScratchDataSizeInBytes = 0;
        }
        break;
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL:
        {
            const UINT numLeaves = NumElements;

            const UINT numInternalNodes = GetNumberOfInternalNodes(numLeaves);
            const UINT totalNumNodes = numLeaves + numInternalNodes;

            pInfo->ResultDataMaxSizeInBytes = sizeof(BVHOffsets) + sizeof(AABBNode) * totalNumNodes + sizeof(BVHMetadata) * numLeaves;
            m_updateAllowed = (pDesc->Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) != 0;
            if (m_updateAllowed)
            {
                pInfo->ResultDataMaxSizeInBytes += numLeaves * sizeof(UINT); // Saved sorted index buffer
                pInfo->ResultDataMaxSizeInBytes += totalNumNodes * sizeof(UINT); // Parent indices for AABBNodes
            }

            pInfo->ScratchDataSizeInBytes = CalculateScratchMemoryUsage(Level::Top, numLeaves).TotalSize;
            pInfo->UpdateScratchDataSizeInBytes = 0;
        }
        break;
        default:
            ThrowFailure(E_INVALIDARG, L"Unrecognized D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE provided");
        }
    }

    void GpuBvh2Builder::EmitRaytracingAccelerationStructurePostBuildInfo(
        _In_  ID3D12GraphicsCommandList *pCommandList,
        _In_  D3D12_GPU_VIRTUAL_ADDRESS_RANGE DestBuffer,
        _In_  UINT NumSourceAccelerationStructures,
        _In_reads_(NumSourceAccelerationStructures)  const D3D12_GPU_VIRTUAL_ADDRESS *pSourceAccelerationStructureData)
    {
        m_postBuildInfoQuery.GetCompactedBVHSizes(
            pCommandList,
            DestBuffer,
            NumSourceAccelerationStructures,
            pSourceAccelerationStructureData);
    }
}
