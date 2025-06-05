using Unity.Jobs;
using UnityEngine.Assertions;
using UnityEngine.U2D.Common;

namespace UnityEngine.U2D.Animation
{
    internal class CpuDeformationSystem : BaseDeformationSystem
    {
        const string k_GpuSkinningShaderKeyword = "SKINNED_SPRITE";
        JobHandle m_BoundJobHandle;
        JobHandle m_CopyJobHandle;

        public override DeformationMethods deformationMethod => DeformationMethods.Cpu;

        internal override void Cleanup()
        {
            base.Cleanup();

            m_BoundJobHandle.Complete();
            m_CopyJobHandle.Complete();
        }

        internal override void UpdateMaterial(SpriteSkin spriteSkin)
        {
            var sharedMaterial = spriteSkin.spriteRenderer.sharedMaterial;
            if (sharedMaterial.IsKeywordEnabled(k_GpuSkinningShaderKeyword))
                sharedMaterial.DisableKeyword(k_GpuSkinningShaderKeyword);
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
            m_CopyJobHandle = ScheduleCopySpriteRendererBuffersJob(jobHandle, batchCount);
            m_BoundJobHandle = ScheduleCalculateSpriteSkinAABBJob(m_DeformJobHandle, batchCount);
            Profiling.scheduleJobs.End();

            JobHandle.ScheduleBatchedJobs();
            jobHandle = JobHandle.CombineDependencies(m_BoundJobHandle, m_CopyJobHandle);
            jobHandle.Complete();

            using (Profiling.setBatchDeformableBufferAndLocalAABB.Auto())
            {
                InternalEngineBridge.SetBatchDeformableBufferAndLocalAABBArray(m_SpriteRenderers, m_Buffers, m_BufferSizes, m_BoundsData);
            }

            foreach (var spriteSkin in m_SpriteSkins)
            {
                var didDeform = m_IsSpriteSkinActiveForDeform[spriteSkin.dataIndex];
                spriteSkin.PostDeform(didDeform);

            }

            DeactivateDeformableBuffers();
        }

        void ResizeBuffers(int vertexBufferSize, in PerSkinJobData skinBatch)
        {
            m_DeformedVerticesBuffer = BufferManager.instance.GetBuffer(m_ObjectId, vertexBufferSize);
            NativeArrayHelpers.ResizeIfNeeded(ref m_FinalBoneTransforms, skinBatch.bindPosesIndex.y);
            NativeArrayHelpers.ResizeIfNeeded(ref m_BoneLookupData, skinBatch.bindPosesIndex.y);
            NativeArrayHelpers.ResizeIfNeeded(ref m_VertexLookupData, skinBatch.verticesIndex.y);
        }
    }
}