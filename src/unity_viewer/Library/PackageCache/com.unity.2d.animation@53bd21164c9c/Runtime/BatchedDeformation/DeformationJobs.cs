using System;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;

namespace UnityEngine.U2D.Animation
{
    internal struct PerSkinJobData
    {
        public int deformVerticesStartPos;
        public int2 bindPosesIndex;
        public int2 verticesIndex;
    }

    internal struct SpriteSkinData
    {
        public NativeCustomSlice<Vector3> vertices;
        public NativeCustomSlice<BoneWeight> boneWeights;
        public NativeCustomSlice<Matrix4x4> bindPoses;
        public NativeCustomSlice<Vector4> tangents;
        public bool hasTangents;
        public int spriteVertexStreamSize;
        public int spriteVertexCount;
        public int tangentVertexOffset;
        public int deformVerticesStartPos;
        public int transformId;
        public NativeCustomSlice<int> boneTransformId;
    }

    [BurstCompile]
    internal struct PrepareDeformJob : IJob
    {
        [ReadOnly]
        public NativeArray<PerSkinJobData> perSkinJobData;
        [ReadOnly]
        public int batchDataSize;
        [WriteOnly]
        public NativeArray<int2> boneLookupData;
        [WriteOnly]
        public NativeArray<int2> vertexLookupData;

        public void Execute()
        {
            for (var i = 0; i < batchDataSize; ++i)
            {
                var jobData = perSkinJobData[i];
                for (int k = 0, j = jobData.bindPosesIndex.x; j < jobData.bindPosesIndex.y; ++j, ++k)
                {
                    boneLookupData[j] = new int2(i, k);
                }

                for (int k = 0, j = jobData.verticesIndex.x; j < jobData.verticesIndex.y; ++j, ++k)
                {
                    vertexLookupData[j] = new int2(i, k);
                }
            }
        }
    }

    [BurstCompile]
    internal struct BoneDeformBatchedJob : IJobParallelFor
    {
        [ReadOnly]
        public NativeArray<float4x4> boneTransform;
        [ReadOnly]
        public NativeArray<float4x4> rootTransform;
        [ReadOnly]
        public NativeArray<int2> boneLookupData;
        [ReadOnly]
        public NativeArray<SpriteSkinData> spriteSkinData;
        [ReadOnly]
        public NativeHashMap<int, TransformAccessJob.TransformData> rootTransformIndex;
        [ReadOnly]
        public NativeHashMap<int, TransformAccessJob.TransformData> boneTransformIndex;
        [WriteOnly]
        public NativeArray<float4x4> finalBoneTransforms;

        public void Execute(int i)
        {
            var x = boneLookupData[i].x;
            var y = boneLookupData[i].y;
            var ssd = spriteSkinData[x];
            var v = ssd.boneTransformId[y];
            var index = boneTransformIndex[v].transformIndex;
            if (index < 0)
                return;
            var aa = boneTransform[index];
            var bb = ssd.bindPoses[y];
            var cc = rootTransformIndex[ssd.transformId].transformIndex;
            finalBoneTransforms[i] = math.mul(rootTransform[cc], math.mul(aa, bb));
        }
    }

    [BurstCompile]
    internal struct SkinDeformBatchedJob : IJobParallelFor
    {
        public NativeSlice<byte> vertices;

        [ReadOnly]
        public NativeArray<float4x4> finalBoneTransforms;
        [ReadOnly]
        public NativeArray<PerSkinJobData> perSkinJobData;
        [ReadOnly]
        public NativeArray<SpriteSkinData> spriteSkinData;
        [ReadOnly]
        public NativeArray<int2> vertexLookupData;

        public unsafe void Execute(int i)
        {
            var j = vertexLookupData[i].x;
            var k = vertexLookupData[i].y;

            var perSkinData = perSkinJobData[j];
            var spriteSkin = spriteSkinData[j];
            var srcVertex = (float3)spriteSkin.vertices[k];
            var tangents = (float4)spriteSkin.tangents[k];
            var influence = spriteSkin.boneWeights[k];

            var bone0 = influence.boneIndex0 + perSkinData.bindPosesIndex.x;
            var bone1 = influence.boneIndex1 + perSkinData.bindPosesIndex.x;
            var bone2 = influence.boneIndex2 + perSkinData.bindPosesIndex.x;
            var bone3 = influence.boneIndex3 + perSkinData.bindPosesIndex.x;

            var deformedPosOffset = (byte*)vertices.GetUnsafePtr();
            var deformedPosStart = deformedPosOffset + spriteSkin.deformVerticesStartPos;
            var deformableVerticesFloat3 = NativeSliceUnsafeUtility.ConvertExistingDataToNativeSlice<float3>(deformedPosStart, spriteSkin.spriteVertexStreamSize, spriteSkin.spriteVertexCount);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeSliceUnsafeUtility.SetAtomicSafetyHandle(ref deformableVerticesFloat3, NativeSliceUnsafeUtility.GetAtomicSafetyHandle(vertices));
#endif
            if (spriteSkin.hasTangents)
            {
                var deformedTanOffset = deformedPosStart + spriteSkin.tangentVertexOffset;
                var deformableTangentsFloat4 = NativeSliceUnsafeUtility.ConvertExistingDataToNativeSlice<float4>(deformedTanOffset, spriteSkin.spriteVertexStreamSize, spriteSkin.spriteVertexCount);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                NativeSliceUnsafeUtility.SetAtomicSafetyHandle(ref deformableTangentsFloat4, NativeSliceUnsafeUtility.GetAtomicSafetyHandle(vertices));
#endif
                var tangent = new float4(tangents.xyz, 0.0f);
                tangent =
                    math.mul(finalBoneTransforms[bone0], tangent) * influence.weight0 +
                    math.mul(finalBoneTransforms[bone1], tangent) * influence.weight1 +
                    math.mul(finalBoneTransforms[bone2], tangent) * influence.weight2 +
                    math.mul(finalBoneTransforms[bone3], tangent) * influence.weight3;
                deformableTangentsFloat4[k] = new float4(math.normalize(tangent.xyz), tangents.w);
            }

            deformableVerticesFloat3[k] =
                math.transform(finalBoneTransforms[bone0], srcVertex) * influence.weight0 +
                math.transform(finalBoneTransforms[bone1], srcVertex) * influence.weight1 +
                math.transform(finalBoneTransforms[bone2], srcVertex) * influence.weight2 +
                math.transform(finalBoneTransforms[bone3], srcVertex) * influence.weight3;
        }
    }

    [BurstCompile]
    internal struct CalculateSpriteSkinAABBJob : IJobParallelFor
    {
        public NativeSlice<byte> vertices;
        [ReadOnly]
        public NativeArray<bool> isSpriteSkinValidForDeformArray;
        [ReadOnly]
        public NativeArray<SpriteSkinData> spriteSkinData;

        [WriteOnly]
        public NativeArray<Bounds> bounds;

        public unsafe void Execute(int i)
        {
            if (!isSpriteSkinValidForDeformArray[i])
                return;

            var spriteSkin = spriteSkinData[i];
            var deformedPosOffset = (byte*)vertices.GetUnsafePtr();
            var deformableVerticesFloat3 = NativeSliceUnsafeUtility.ConvertExistingDataToNativeSlice<float3>(deformedPosOffset + spriteSkin.deformVerticesStartPos, spriteSkin.spriteVertexStreamSize, spriteSkin.spriteVertexCount);

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeSliceUnsafeUtility.SetAtomicSafetyHandle(ref deformableVerticesFloat3, NativeSliceUnsafeUtility.GetAtomicSafetyHandle(vertices));
#endif

            bounds[i] = SpriteSkinUtility.CalculateSpriteSkinBounds(deformableVerticesFloat3);
        }
    }

    [BurstCompile]
    internal struct FillPerSkinJobSingleThread : IJob
    {
        public PerSkinJobData combinedSkinBatch;

        [ReadOnly]
        public NativeArray<bool> isSpriteSkinValidForDeformArray;

        public NativeArray<SpriteSkinData> spriteSkinDataArray;
        public NativeArray<PerSkinJobData> perSkinJobDataArray;

        public NativeArray<PerSkinJobData> combinedSkinBatchArray;

        public void Execute()
        {
            var startIndex = 0;
            var endIndex = spriteSkinDataArray.Length;
            for (var index = startIndex; index < endIndex; ++index)
            {
                var spriteSkinData = spriteSkinDataArray[index];
                spriteSkinData.deformVerticesStartPos = -1;
                var vertexBufferSize = 0;
                var vertexCount = 0;
                var bindPoseCount = 0;
                if (isSpriteSkinValidForDeformArray[index])
                {
                    spriteSkinData.deformVerticesStartPos = combinedSkinBatch.deformVerticesStartPos;
                    vertexBufferSize = spriteSkinData.spriteVertexCount * spriteSkinData.spriteVertexStreamSize;
                    vertexCount = spriteSkinData.spriteVertexCount;
                    bindPoseCount = spriteSkinData.bindPoses.Length;
                }

                combinedSkinBatch.verticesIndex.x = combinedSkinBatch.verticesIndex.y;
                combinedSkinBatch.verticesIndex.y = combinedSkinBatch.verticesIndex.x + vertexCount;
                combinedSkinBatch.bindPosesIndex.x = combinedSkinBatch.bindPosesIndex.y;
                combinedSkinBatch.bindPosesIndex.y = combinedSkinBatch.bindPosesIndex.x + bindPoseCount;
                spriteSkinDataArray[index] = spriteSkinData;
                perSkinJobDataArray[index] = combinedSkinBatch;
                combinedSkinBatch.deformVerticesStartPos += vertexBufferSize;
            }

            combinedSkinBatchArray[0] = combinedSkinBatch;
        }
    }

    [BurstCompile]
    internal struct CopySpriteRendererBuffersJob : IJobParallelFor
    {
        [ReadOnly]
        public NativeArray<bool> isSpriteSkinValidForDeformArray;
        [ReadOnly]
        public NativeArray<SpriteSkinData> spriteSkinData;
        [ReadOnly, NativeDisableUnsafePtrRestriction]
        public IntPtr ptrVertices;

        [WriteOnly]
        public NativeArray<IntPtr> buffers;
        [WriteOnly]
        public NativeArray<int> bufferSizes;

        public void Execute(int i)
        {
            var skinData = spriteSkinData[i];
            var startVertices = default(IntPtr);
            var vertexBufferLength = 0;
            if (isSpriteSkinValidForDeformArray[i])
            {
                startVertices = ptrVertices + skinData.deformVerticesStartPos;
                vertexBufferLength = skinData.spriteVertexCount * skinData.spriteVertexStreamSize;
            }

            buffers[i] = startVertices;
            bufferSizes[i] = vertexBufferLength;
        }
    }

    [BurstCompile]
    internal struct CopySpriteRendererBoneTransformBuffersJob : IJobParallelFor
    {
        [ReadOnly]
        public NativeArray<bool> isSpriteSkinValidForDeformArray;
        [ReadOnly]
        public NativeArray<SpriteSkinData> spriteSkinData;
        [ReadOnly]
        public NativeArray<PerSkinJobData> perSkinJobData;

        [ReadOnly, NativeDisableUnsafePtrRestriction]
        public IntPtr ptrBoneTransforms;

        [WriteOnly]
        public NativeArray<IntPtr> buffers;
        [WriteOnly]
        public NativeArray<int> bufferSizes;

        public void Execute(int i)
        {
            var skinData = spriteSkinData[i];
            var skinJobData = perSkinJobData[i];
            var startMatrix = default(IntPtr);
            var matrixLength = 0;
            if (isSpriteSkinValidForDeformArray[i])
            {
                startMatrix = ptrBoneTransforms + (skinJobData.bindPosesIndex.x * 64);
                matrixLength = skinData.boneTransformId.Length;
            }

            buffers[i] = startMatrix;
            bufferSizes[i] = matrixLength;
        }
    }
}
