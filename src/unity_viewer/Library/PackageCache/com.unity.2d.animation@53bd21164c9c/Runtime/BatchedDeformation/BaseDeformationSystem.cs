using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine.U2D.Common;

namespace UnityEngine.U2D.Animation
{
    internal abstract class BaseDeformationSystem
    {
        protected static class Profiling
        {
            public static readonly ProfilerMarker transformAccessJob = new ProfilerMarker("BaseDeformationSystem.TransformAccessJob");
            public static readonly ProfilerMarker getSpriteSkinBatchData = new ProfilerMarker("BaseDeformationSystem.GetSpriteSkinBatchData");
            public static readonly ProfilerMarker scheduleJobs = new ProfilerMarker("BaseDeformationSystem.ScheduleJobs");
            public static readonly ProfilerMarker setBatchDeformableBufferAndLocalAABB = new ProfilerMarker("BaseDeformationSystem.SetBatchDeformableBufferAndLocalAABB");
            public static readonly ProfilerMarker setBoneTransformsArray = new ProfilerMarker("BaseDeformationSystem.SetBoneTransformsArray");
        }

        public abstract DeformationMethods deformationMethod { get; }

        protected int m_ObjectId;

        protected readonly HashSet<SpriteSkin> m_SpriteSkins = new HashSet<SpriteSkin>();
        protected SpriteRenderer[] m_SpriteRenderers = new SpriteRenderer[0];

        readonly HashSet<SpriteSkin> m_SpriteSkinsToAdd = new HashSet<SpriteSkin>();
        readonly HashSet<SpriteSkin> m_SpriteSkinsToRemove = new HashSet<SpriteSkin>();
        readonly List<int> m_TransformIdsToRemove = new List<int>();

        protected NativeByteArray m_DeformedVerticesBuffer;
        protected NativeArray<float4x4> m_FinalBoneTransforms;

        protected NativeArray<bool> m_IsSpriteSkinActiveForDeform;
        protected NativeArray<SpriteSkinData> m_SpriteSkinData;
        protected NativeArray<PerSkinJobData> m_PerSkinJobData;
        protected NativeArray<Bounds> m_BoundsData;
        protected NativeArray<IntPtr> m_Buffers;
        protected NativeArray<int> m_BufferSizes;
        protected NativeArray<IntPtr> m_BoneTransformBuffers;

        protected NativeArray<int2> m_BoneLookupData;
        protected NativeArray<int2> m_VertexLookupData;
        protected NativeArray<PerSkinJobData> m_SkinBatchArray;

        protected TransformAccessJob m_LocalToWorldTransformAccessJob;
        protected TransformAccessJob m_WorldToLocalTransformAccessJob;

        protected JobHandle m_DeformJobHandle;

        internal void RemoveBoneTransforms(SpriteSkin spriteSkin)
        {
            if (!m_SpriteSkins.Contains(spriteSkin))
                return;

            m_LocalToWorldTransformAccessJob.RemoveTransformById(spriteSkin.rootBoneTransformId);
            var boneTransforms = spriteSkin.boneTransformId;
            if (boneTransforms == default || !boneTransforms.IsCreated)
                return;

            for (var i = 0; i < boneTransforms.Length; ++i)
                m_LocalToWorldTransformAccessJob.RemoveTransformById(boneTransforms[i]);
        }

        internal void AddBoneTransforms(SpriteSkin spriteSkin)
        {
            if (!m_SpriteSkins.Contains(spriteSkin))
                return;

            m_LocalToWorldTransformAccessJob.AddTransform(spriteSkin.rootBone);
            if (spriteSkin.boneTransforms != null)
            {
                foreach (var t in spriteSkin.boneTransforms)
                {
                    if (t != null)
                        m_LocalToWorldTransformAccessJob.AddTransform(t);
                }
            }
        }

        internal virtual void UpdateMaterial(SpriteSkin spriteSkin) { }

        internal virtual bool AddSpriteSkin(SpriteSkin spriteSkin)
        {
            if (!m_SpriteSkins.Contains(spriteSkin) && !m_SpriteSkinsToAdd.Contains(spriteSkin))
            {
                m_SpriteSkinsToAdd.Add(spriteSkin);
                return true;
            }

            if (m_SpriteSkinsToRemove.Contains(spriteSkin))
            {
                m_SpriteSkinsToAdd.Add(spriteSkin);
                return true;
            }

            return false;
        }

        internal void CopyToSpriteSkinData(SpriteSkin spriteSkin)
        {
            if (!m_SpriteSkinData.IsCreated)
                throw new InvalidOperationException("Sprite Skin Data not initialized.");

            var dataIndex = spriteSkin.dataIndex;
            if(dataIndex < 0 || dataIndex >= m_SpriteSkinData.Length)
                return;

            var spriteSkinData = default(SpriteSkinData);
            spriteSkin.CopyToSpriteSkinData(ref spriteSkinData);

            m_SpriteSkinData[dataIndex] = spriteSkinData;
            m_SpriteRenderers[dataIndex] = spriteSkin.spriteRenderer;
        }

        internal void RemoveSpriteSkin(SpriteSkin spriteSkin)
        {
            if (spriteSkin == null)
                return;

            if (m_SpriteSkins.Contains(spriteSkin) && !m_SpriteSkinsToRemove.Contains(spriteSkin))
            {
                m_SpriteSkinsToRemove.Add(spriteSkin);
                m_TransformIdsToRemove.Add(spriteSkin.transform.GetInstanceID());
            }

            if (m_SpriteSkinsToAdd.Contains(spriteSkin))
                m_SpriteSkinsToAdd.Remove(spriteSkin);

            RemoveBoneTransforms(spriteSkin);
        }

        internal HashSet<SpriteSkin> GetSpriteSkins()
        {
            return m_SpriteSkins;
        }

        internal void Initialize(int objectId)
        {
            m_ObjectId = objectId;

            if (m_LocalToWorldTransformAccessJob == null)
                m_LocalToWorldTransformAccessJob = new TransformAccessJob();
            if (m_WorldToLocalTransformAccessJob == null)
                m_WorldToLocalTransformAccessJob = new TransformAccessJob();

            InitializeArrays();
            BatchRemoveSpriteSkins();
            BatchAddSpriteSkins();

            // Initialise all existing SpriteSkins as execution order is indeterminate
            var count = 0;
            foreach (var spriteSkin in m_SpriteSkins)
            {
                spriteSkin.SetDataIndex(count++);

                CopyToSpriteSkinData(spriteSkin);
            }
        }

        protected virtual void InitializeArrays()
        {
            const int startingCount = 0;

            m_FinalBoneTransforms = new NativeArray<float4x4>(startingCount, Allocator.Persistent);
            m_BoneLookupData = new NativeArray<int2>(startingCount, Allocator.Persistent);
            m_VertexLookupData = new NativeArray<int2>(startingCount, Allocator.Persistent);
            m_SkinBatchArray = new NativeArray<PerSkinJobData>(startingCount, Allocator.Persistent);

            m_IsSpriteSkinActiveForDeform = new NativeArray<bool>(startingCount, Allocator.Persistent);
            m_PerSkinJobData = new NativeArray<PerSkinJobData>(startingCount, Allocator.Persistent);
            m_SpriteSkinData = new NativeArray<SpriteSkinData>(startingCount, Allocator.Persistent);
            m_BoundsData = new NativeArray<Bounds>(startingCount, Allocator.Persistent);
            m_Buffers = new NativeArray<IntPtr>(startingCount, Allocator.Persistent);
            m_BufferSizes = new NativeArray<int>(startingCount, Allocator.Persistent);
        }

        protected void BatchRemoveSpriteSkins()
        {
            var spritesToRemoveCount = m_SpriteSkinsToRemove.Count;
            if (spritesToRemoveCount == 0)
                return;

            m_WorldToLocalTransformAccessJob.RemoveTransformsByIds(m_TransformIdsToRemove);

            var updatedCount = Math.Max(m_SpriteSkins.Count - spritesToRemoveCount, 0);
            if (updatedCount == 0)
            {
                m_SpriteSkins.Clear();
            }
            else
            {
                foreach (var spriteSkin in m_SpriteSkinsToRemove)
                    m_SpriteSkins.Remove(spriteSkin);
            }

            var count = 0;
            foreach (var spriteSkin in m_SpriteSkins)
            {
                spriteSkin.SetDataIndex(count++);
                CopyToSpriteSkinData(spriteSkin);
            }

            Array.Resize(ref m_SpriteRenderers, updatedCount);
            ResizeAndCopyArrays(updatedCount);

            m_TransformIdsToRemove.Clear();
            m_SpriteSkinsToRemove.Clear();
        }

        protected void BatchAddSpriteSkins()
        {
            if (m_SpriteSkinsToAdd.Count == 0)
                return;

            if (!m_IsSpriteSkinActiveForDeform.IsCreated)
                throw new InvalidOperationException("SpriteSkinActiveForDeform not initialized.");

            var updatedCount = m_SpriteSkins.Count + m_SpriteSkinsToAdd.Count;
            Array.Resize(ref m_SpriteRenderers, updatedCount);
            ResizeAndCopyArrays(updatedCount);

            foreach (var spriteSkin in m_SpriteSkinsToAdd)
            {
                if (m_SpriteSkins.Contains(spriteSkin))
                {
                    Debug.LogError($"Skin already exists! Name={spriteSkin.name}");
                    continue;
                }

                m_SpriteSkins.Add(spriteSkin);
                UpdateMaterial(spriteSkin);
                var count = m_SpriteSkins.Count;

                m_SpriteRenderers[count - 1] = spriteSkin.spriteRenderer;
                m_WorldToLocalTransformAccessJob.AddTransform(spriteSkin.transform);

                AddBoneTransforms(spriteSkin);

                spriteSkin.SetDataIndex(count - 1);
                CopyToSpriteSkinData(spriteSkin);
            }

            m_SpriteSkinsToAdd.Clear();
        }

        protected virtual void ResizeAndCopyArrays(int updatedCount)
        {
            NativeArrayHelpers.ResizeAndCopyIfNeeded(ref m_IsSpriteSkinActiveForDeform, updatedCount);
            NativeArrayHelpers.ResizeAndCopyIfNeeded(ref m_PerSkinJobData, updatedCount);
            NativeArrayHelpers.ResizeAndCopyIfNeeded(ref m_Buffers, updatedCount);
            NativeArrayHelpers.ResizeAndCopyIfNeeded(ref m_BufferSizes, updatedCount);
            NativeArrayHelpers.ResizeAndCopyIfNeeded(ref m_SpriteSkinData, updatedCount);
            NativeArrayHelpers.ResizeAndCopyIfNeeded(ref m_BoundsData, updatedCount);
        }

        internal virtual void Cleanup()
        {
            m_DeformJobHandle.Complete();

            m_SpriteSkins.Clear();
            m_SpriteRenderers = new SpriteRenderer[0];
            BufferManager.instance.ReturnBuffer(m_ObjectId);
            m_IsSpriteSkinActiveForDeform.DisposeIfCreated();
            m_PerSkinJobData.DisposeIfCreated();
            m_Buffers.DisposeIfCreated();
            m_BufferSizes.DisposeIfCreated();
            m_SpriteSkinData.DisposeIfCreated();
            m_BoneLookupData.DisposeIfCreated();
            m_VertexLookupData.DisposeIfCreated();
            m_SkinBatchArray.DisposeIfCreated();
            m_FinalBoneTransforms.DisposeIfCreated();
            m_BoundsData.DisposeIfCreated();

            m_LocalToWorldTransformAccessJob.Destroy();
            m_WorldToLocalTransformAccessJob.Destroy();
        }

        internal abstract void Update();

        protected void PrepareDataForDeformation(out JobHandle localToWorldJobHandle, out JobHandle worldToLocalJobHandle)
        {
            ValidateSpriteSkinData();

            using (Profiling.transformAccessJob.Auto())
            {
                localToWorldJobHandle = m_LocalToWorldTransformAccessJob.StartLocalToWorldJob();
                worldToLocalJobHandle = m_WorldToLocalTransformAccessJob.StartWorldToLocalJob();
            }

            using (Profiling.getSpriteSkinBatchData.Auto())
            {
                NativeArrayHelpers.ResizeIfNeeded(ref m_SkinBatchArray, 1);
                var fillPerSkinJobSingleThread = new FillPerSkinJobSingleThread()
                {
                    isSpriteSkinValidForDeformArray = m_IsSpriteSkinActiveForDeform,
                    combinedSkinBatchArray = m_SkinBatchArray,
                    spriteSkinDataArray = m_SpriteSkinData,
                    perSkinJobDataArray = m_PerSkinJobData,
                };
                fillPerSkinJobSingleThread.Run();
            }
        }

        void ValidateSpriteSkinData()
        {
            foreach (var spriteSkin in m_SpriteSkins)
            {
                var index = spriteSkin.dataIndex;
                m_IsSpriteSkinActiveForDeform[index] = spriteSkin.BatchValidate();
                if (m_IsSpriteSkinActiveForDeform[index] && spriteSkin.NeedToUpdateDeformationCache())
                    CopyToSpriteSkinData(spriteSkin);
            }
        }

        protected bool GotVerticesToDeform(out int vertexBufferSize)
        {
            vertexBufferSize = m_SkinBatchArray[0].deformVerticesStartPos;
            return vertexBufferSize > 0;
        }

        protected JobHandle SchedulePrepareJob(int batchCount)
        {
            var prepareJob = new PrepareDeformJob
            {
                batchDataSize = batchCount,
                perSkinJobData = m_PerSkinJobData,
                boneLookupData = m_BoneLookupData,
                vertexLookupData = m_VertexLookupData
            };
            return prepareJob.Schedule();
        }

        protected JobHandle ScheduleBoneJobBatched(JobHandle jobHandle, PerSkinJobData skinBatch)
        {
            var boneJobBatched = new BoneDeformBatchedJob()
            {
                boneTransform = m_LocalToWorldTransformAccessJob.transformMatrix,
                rootTransform = m_WorldToLocalTransformAccessJob.transformMatrix,
                spriteSkinData = m_SpriteSkinData,
                boneLookupData = m_BoneLookupData,
                finalBoneTransforms = m_FinalBoneTransforms,
                rootTransformIndex = m_WorldToLocalTransformAccessJob.transformData,
                boneTransformIndex = m_LocalToWorldTransformAccessJob.transformData
            };
            jobHandle = boneJobBatched.Schedule(skinBatch.bindPosesIndex.y, 8, jobHandle);
            return jobHandle;
        }

        protected JobHandle ScheduleSkinDeformBatchedJob(JobHandle jobHandle, PerSkinJobData skinBatch)
        {
            var skinJobBatched = new SkinDeformBatchedJob()
            {
                vertices = m_DeformedVerticesBuffer.array,
                vertexLookupData = m_VertexLookupData,
                spriteSkinData = m_SpriteSkinData,
                perSkinJobData = m_PerSkinJobData,
                finalBoneTransforms = m_FinalBoneTransforms,
            };
            return skinJobBatched.Schedule(skinBatch.verticesIndex.y, 16, jobHandle);
        }

        protected unsafe JobHandle ScheduleCopySpriteRendererBuffersJob(JobHandle jobHandle, int batchCount)
        {
            var copySpriteRendererBuffersJob = new CopySpriteRendererBuffersJob()
            {
                isSpriteSkinValidForDeformArray = m_IsSpriteSkinActiveForDeform,
                spriteSkinData = m_SpriteSkinData,
                ptrVertices = (IntPtr)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(m_DeformedVerticesBuffer.array),
                buffers = m_Buffers,
                bufferSizes = m_BufferSizes,
            };
            return copySpriteRendererBuffersJob.Schedule(batchCount, 16, jobHandle);
        }

        protected JobHandle ScheduleCalculateSpriteSkinAABBJob(JobHandle jobHandle, int batchCount)
        {
            var updateBoundJob = new CalculateSpriteSkinAABBJob
            {
                vertices = m_DeformedVerticesBuffer.array,
                isSpriteSkinValidForDeformArray = m_IsSpriteSkinActiveForDeform,
                spriteSkinData = m_SpriteSkinData,
                bounds = m_BoundsData,
            };
            return updateBoundJob.Schedule(batchCount, 4, jobHandle);
        }

        protected void DeactivateDeformableBuffers()
        {
            for (var i = 0; i < m_IsSpriteSkinActiveForDeform.Length; ++i)
            {
                if (m_IsSpriteSkinActiveForDeform[i] || InternalEngineBridge.IsUsingDeformableBuffer(m_SpriteRenderers[i], IntPtr.Zero))
                    continue;
                m_SpriteRenderers[i].DeactivateDeformableBuffer();
            }
        }

        internal bool IsSpriteSkinActiveForDeformation(SpriteSkin spriteSkin)
        {
            return m_IsSpriteSkinActiveForDeform[spriteSkin.dataIndex];
        }

        internal unsafe NativeArray<byte> GetDeformableBufferForSpriteSkin(SpriteSkin spriteSkin)
        {
            if (!m_SpriteSkins.Contains(spriteSkin))
                return default;

            if (!m_DeformJobHandle.IsCompleted)
                m_DeformJobHandle.Complete();

            var skinData = m_SpriteSkinData[spriteSkin.dataIndex];
            if (skinData.deformVerticesStartPos < 0)
                return default;

            var vertexBufferLength = skinData.spriteVertexCount * skinData.spriteVertexStreamSize;
            var ptrVertices = (byte*)m_DeformedVerticesBuffer.array.GetUnsafeReadOnlyPtr();
            ptrVertices += skinData.deformVerticesStartPos;
            var buffer = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<byte>(ptrVertices, vertexBufferLength, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref buffer, NativeArrayUnsafeUtility.GetAtomicSafetyHandle(m_DeformedVerticesBuffer.array));
#endif
            return buffer;
        }

#if UNITY_INCLUDE_TESTS
        internal TransformAccessJob GetWorldToLocalTransformAccessJob() => m_WorldToLocalTransformAccessJob;
        internal TransformAccessJob GetLocalToWorldTransformAccessJob() => m_LocalToWorldTransformAccessJob;
#endif
    }
}
