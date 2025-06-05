using System;
using System.Collections.Generic;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.U2D.Common;
using UnityEngine.Assertions;

namespace UnityEngine.U2D.Animation
{
    internal class GpuDeformationSystem : BaseDeformationSystem
    {
        const string k_GpuSkinningShaderKeyword = "SKINNED_SPRITE";
        const string k_GlobalSpriteBoneBufferId = "_SpriteBoneTransforms";

        readonly Dictionary<int, Material> m_KeywordEnabledMaterials = new Dictionary<int, Material>();

        NativeArray<int> m_BoneTransformBufferSizes;
        ComputeBuffer m_BoneTransformsComputeBuffer;
        static ComputeBuffer s_FallbackBuffer;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void CreateFallbackBuffer()
        {
            if (s_FallbackBuffer == null)
                s_FallbackBuffer = new ComputeBuffer(UnsafeUtility.SizeOf<float4x4>(), UnsafeUtility.SizeOf<float4x4>(), ComputeBufferType.Default);

            Shader.SetGlobalBuffer(k_GlobalSpriteBoneBufferId, s_FallbackBuffer);
        }

        static void ClearFallbackBuffer()
        {
            if(s_FallbackBuffer != null)
                s_FallbackBuffer.Release();

            s_FallbackBuffer = null;
        }

        public override DeformationMethods deformationMethod => DeformationMethods.Gpu;

        internal static bool DoesShaderSupportGpuDeformation(Material material)
        {
            if (material == null)
                return false;
            var shader = material.shader;
            if (shader == null)
                return false;

            var supportedKeywords = shader.keywordSpace.keywords;
            for (var i = 0; i < supportedKeywords.Length; ++i)
            {
                if (supportedKeywords[i].name == k_GpuSkinningShaderKeyword)
                    return true;
            }

            return false;
        }

        static bool IsComputeBufferValid(ComputeBuffer buffer) => buffer != null && buffer.IsValid();

        protected override void InitializeArrays()
        {
            base.InitializeArrays();

            const int startingCount = 0;
            m_BoneTransformBuffers = new NativeArray<IntPtr>(startingCount, Allocator.Persistent);
            m_BoneTransformBufferSizes = new NativeArray<int>(startingCount, Allocator.Persistent);

            CreateFallbackBuffer();
        }

        internal override void Cleanup()
        {
            base.Cleanup();

            m_BoneTransformBuffers.DisposeIfCreated();
            m_BoneTransformBufferSizes.DisposeIfCreated();

            CleanupComputeResources();

            ClearFallbackBuffer();
        }

        protected override void ResizeAndCopyArrays(int updatedCount)
        {
            base.ResizeAndCopyArrays(updatedCount);
            NativeArrayHelpers.ResizeAndCopyIfNeeded(ref m_BoneTransformBuffers, updatedCount);
            NativeArrayHelpers.ResizeAndCopyIfNeeded(ref m_BoneTransformBufferSizes, updatedCount);

            if (updatedCount == 0)
                CleanupComputeResources();
        }

        void CleanupComputeResources()
        {
            if (IsComputeBufferValid(m_BoneTransformsComputeBuffer))
                m_BoneTransformsComputeBuffer.Release();
            m_BoneTransformsComputeBuffer = null;

            foreach (var material in m_KeywordEnabledMaterials.Values)
                material.DisableKeyword(k_GpuSkinningShaderKeyword);
            m_KeywordEnabledMaterials.Clear();
            Shader.SetGlobalBuffer(k_GlobalSpriteBoneBufferId, s_FallbackBuffer);
        }

        internal override void UpdateMaterial(SpriteSkin spriteSkin)
        {
            var sharedMaterial = spriteSkin.spriteRenderer.sharedMaterial;
            if (!sharedMaterial.IsKeywordEnabled(k_GpuSkinningShaderKeyword))
                sharedMaterial.EnableKeyword(k_GpuSkinningShaderKeyword);
        }

        internal override bool AddSpriteSkin(SpriteSkin spriteSkin)
        {
            var success = base.AddSpriteSkin(spriteSkin);

            var sharedMaterial = spriteSkin.spriteRenderer.sharedMaterial;
            if (!sharedMaterial.IsKeywordEnabled(k_GpuSkinningShaderKeyword))
            {
                sharedMaterial.EnableKeyword(k_GpuSkinningShaderKeyword);
                m_KeywordEnabledMaterials.TryAdd(sharedMaterial.GetInstanceID(), sharedMaterial);
            }

            return success;
        }

        internal override void Update()
        {
            BatchRemoveSpriteSkins();
            BatchAddSpriteSkins();

            var count = m_SpriteSkins.Count;
            if (count == 0)
            {
                m_LocalToWorldTransformAccessJob.ResetCache();
                m_WorldToLocalTransformAccessJob.ResetCache();
                return;
            }

            Assert.AreEqual(m_IsSpriteSkinActiveForDeform.Length, count);
            Assert.AreEqual(m_PerSkinJobData.Length, count);
            Assert.AreEqual(m_SpriteSkinData.Length, count);
            Assert.AreEqual(m_BoundsData.Length, count);
            Assert.AreEqual(m_SpriteRenderers.Length, count);
            Assert.AreEqual(m_Buffers.Length, count);
            Assert.AreEqual(m_BufferSizes.Length, count);

            Assert.AreEqual(m_BoneTransformBuffers.Length, count);
            Assert.AreEqual(m_BoneTransformBufferSizes.Length, count);

            PrepareDataForDeformation(out var localToWorldJobHandle, out var worldToLocalJobHandle);

            if (!GotVerticesToDeform(out var vertexBufferSize))
            {
                localToWorldJobHandle.Complete();
                worldToLocalJobHandle.Complete();
                DeactivateDeformableBuffers();
                return;
            }

            var skinBatch = m_SkinBatchArray[0];
            ResizeBuffers(vertexBufferSize, in skinBatch);

            var batchCount = m_SpriteSkinData.Length;
            var jobHandle = SchedulePrepareJob(batchCount);

            Profiling.scheduleJobs.Begin();
            jobHandle = JobHandle.CombineDependencies(localToWorldJobHandle, worldToLocalJobHandle, jobHandle);
            jobHandle = ScheduleBoneJobBatched(jobHandle, skinBatch);
            m_DeformJobHandle = ScheduleSkinDeformBatchedJob(jobHandle, skinBatch);
            jobHandle = ScheduleCopySpriteRendererBuffersJob(m_DeformJobHandle, batchCount);
            jobHandle = ScheduleCopySpriteRendererBoneTransformBuffersJob(jobHandle, batchCount);
            jobHandle = ScheduleCalculateSpriteSkinAABBJob(jobHandle, batchCount);
            Profiling.scheduleJobs.End();

            JobHandle.ScheduleBatchedJobs();
            jobHandle.Complete();

            using (Profiling.setBoneTransformsArray.Auto())
            {
                InternalEngineBridge.SetBatchBoneTransformsAABBArray(m_SpriteRenderers, m_BoneTransformBuffers, m_BoneTransformBufferSizes, m_BoundsData);
            }

            SetComputeBuffer();

            foreach (var spriteSkin in m_SpriteSkins)
            {
                var didDeform = m_IsSpriteSkinActiveForDeform[spriteSkin.dataIndex];
                spriteSkin.PostDeform(didDeform);
            }

            DeactivateDeformableBuffers();
        }

        void ResizeBuffers(int vertexBufferSize, in PerSkinJobData skinBatch)
        {
            var noOfBones = skinBatch.bindPosesIndex.y;
            var noOfVerticesInBatch = skinBatch.verticesIndex.y;

            m_DeformedVerticesBuffer = BufferManager.instance.GetBuffer(m_ObjectId, vertexBufferSize);
            NativeArrayHelpers.ResizeIfNeeded(ref m_FinalBoneTransforms, noOfBones);
            NativeArrayHelpers.ResizeIfNeeded(ref m_BoneLookupData, noOfBones);
            NativeArrayHelpers.ResizeIfNeeded(ref m_VertexLookupData, noOfVerticesInBatch);

            if (!IsComputeBufferValid(m_BoneTransformsComputeBuffer) || m_BoneTransformsComputeBuffer.count < noOfBones)
                CreateComputeBuffer(noOfBones);
        }

        void CreateComputeBuffer(int bufferSize)
        {
            if (IsComputeBufferValid(m_BoneTransformsComputeBuffer))
                m_BoneTransformsComputeBuffer.Release();

            m_BoneTransformsComputeBuffer = new ComputeBuffer(bufferSize, UnsafeUtility.SizeOf<float4x4>(), ComputeBufferType.Default);
            SetComputeBuffer();
        }

        void SetComputeBuffer()
        {
            m_BoneTransformsComputeBuffer.SetData(m_FinalBoneTransforms, 0, 0, m_FinalBoneTransforms.Length);
            Shader.SetGlobalBuffer(k_GlobalSpriteBoneBufferId, m_BoneTransformsComputeBuffer);
        }

        unsafe JobHandle ScheduleCopySpriteRendererBoneTransformBuffersJob(JobHandle jobHandle, int batchCount)
        {
            var copySpriteRendererBoneTransformBuffersJob = new CopySpriteRendererBoneTransformBuffersJob()
            {
                isSpriteSkinValidForDeformArray = m_IsSpriteSkinActiveForDeform,
                spriteSkinData = m_SpriteSkinData,
                ptrBoneTransforms = (IntPtr)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(m_FinalBoneTransforms),
                perSkinJobData = m_PerSkinJobData,
                buffers = m_BoneTransformBuffers,
                bufferSizes = m_BoneTransformBufferSizes,
            };
            return copySpriteRendererBoneTransformBuffersJob.Schedule(batchCount, 16, jobHandle);
        }
    }
}
