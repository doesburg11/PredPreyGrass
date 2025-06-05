using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal enum WeightEditorMode
    {
        AddAndSubtract,
        GrowAndShrink,
        Smooth
    }

    internal class WeightEditor
    {
        public BaseSpriteMeshData spriteMeshData
        {
            get => m_SpriteMeshDataController.spriteMeshData;
            set => m_SpriteMeshDataController.spriteMeshData = value;
        }

        public ICacheUndo cacheUndo { get; set; }
        public WeightEditorMode mode { get; set; }
        public int boneIndex { get; set; }
        public ISelection<int> selection { get; set; }
        public bool emptySelectionEditsAll { get; set; }
        public bool autoNormalize { get; set; }

        WeightEditorMode currentMode { get; set; }
        bool useRelativeValues { get; set; }

        SpriteMeshDataController m_SpriteMeshDataController = new SpriteMeshDataController();
        const int k_MaxSmoothIterations = 8;
        float[] m_SmoothValues;
        readonly List<BoneWeight[]> m_SmoothedBoneWeights = new List<BoneWeight[]>();
        readonly List<BoneWeight> m_StoredBoneWeights = new List<BoneWeight>();
        int boneCount => spriteMeshData != null ? spriteMeshData.boneCount : 0;

        public WeightEditor()
        {
            autoNormalize = true;
        }

        public void OnEditStart(bool relative)
        {
            Validate();

            RegisterUndo();
            currentMode = mode;
            useRelativeValues = relative;

            if (!useRelativeValues)
                StoreBoneWeights();

            if (mode == WeightEditorMode.Smooth)
                PrepareSmoothingBuffers();
        }

        public void OnEditEnd()
        {
            Validate();

            if (currentMode == WeightEditorMode.AddAndSubtract)
            {
                for (int i = 0; i < spriteMeshData.vertexCount; ++i)
                    spriteMeshData.vertexWeights[i].Clamp(4);
            }

            if (autoNormalize)
                m_SpriteMeshDataController.NormalizeWeights(null);

            m_SpriteMeshDataController.SortTrianglesByDepth();
        }

        public void DoEdit(float value)
        {
            Validate();

            if (!useRelativeValues)
                RestoreBoneWeights();

            if (currentMode == WeightEditorMode.AddAndSubtract)
                SetWeight(value);
            else if (currentMode == WeightEditorMode.GrowAndShrink)
                SetWeight(value, false);
            else if (currentMode == WeightEditorMode.Smooth)
                SmoothWeights(value);
        }

        void Validate()
        {
            if (spriteMeshData == null)
                throw (new Exception(TextContent.noSpriteSelected));
        }

        void RegisterUndo()
        {
            Debug.Assert(cacheUndo != null);

            cacheUndo.BeginUndoOperation(TextContent.editWeights);
        }

        void SetWeight(float value, bool createNewChannel = true)
        {
            if (boneIndex == -1 || spriteMeshData == null)
                return;

            Debug.Assert(selection != null);

            for (var i = 0; i < spriteMeshData.vertexCount; ++i)
            {
                if (selection.Count == 0 && emptySelectionEditsAll ||
                    selection.Count > 0 && selection.Contains(i))
                {
                    var editableBoneWeight = spriteMeshData.vertexWeights[i];
                    var channel = editableBoneWeight.GetChannelFromBoneIndex(boneIndex);

                    if (channel == -1)
                    {
                        if (createNewChannel && value > 0f)
                        {
                            editableBoneWeight.AddChannel(boneIndex, 0f, true);
                            channel = editableBoneWeight.GetChannelFromBoneIndex(boneIndex);
                        }
                        else
                        {
                            continue;
                        }
                    }

                    editableBoneWeight[channel].weight += value;

                    if (editableBoneWeight.Sum() > 1f)
                        editableBoneWeight.CompensateOtherChannels(channel);

                    editableBoneWeight.FilterChannels(0f);
                }
            }
        }

        void SmoothWeights(float value)
        {
            Debug.Assert(selection != null);

            for (var i = 0; i < spriteMeshData.vertexCount; ++i)
            {
                if (selection.Count == 0 && emptySelectionEditsAll ||
                    selection.Count > 0 && selection.Contains(i))
                {
                    var smoothValue = m_SmoothValues[i];

                    if (smoothValue >= k_MaxSmoothIterations)
                        continue;

                    m_SmoothValues[i] = Mathf.Clamp(smoothValue + value, 0f, k_MaxSmoothIterations);

                    var lerpValue = GetLerpValue(m_SmoothValues[i]);
                    var lerpIndex = GetLerpIndex(m_SmoothValues[i]);
                    var smoothedBoneWeightsFloor = GetSmoothedBoneWeights(lerpIndex - 1);
                    var smoothedBoneWeightsCeil = GetSmoothedBoneWeights(lerpIndex);

                    var boneWeight = EditableBoneWeightUtility.Lerp(smoothedBoneWeightsFloor[i], smoothedBoneWeightsCeil[i], lerpValue);
                    spriteMeshData.vertexWeights[i].SetFromBoneWeight(boneWeight);
                }
            }
        }

        void PrepareSmoothingBuffers()
        {
            if (m_SmoothValues == null || m_SmoothValues.Length != spriteMeshData.vertexCount)
                m_SmoothValues = new float[spriteMeshData.vertexCount];

            Array.Clear(m_SmoothValues, 0, m_SmoothValues.Length);

            m_SmoothedBoneWeights.Clear();

            var boneWeights = new BoneWeight[spriteMeshData.vertexCount];

            for (var i = 0; i < spriteMeshData.vertexCount; i++)
            {
                var editableBoneWeight = spriteMeshData.vertexWeights[i];
                boneWeights[i] = editableBoneWeight.ToBoneWeight(false);
            }

            m_SmoothedBoneWeights.Add(boneWeights);
        }

        BoneWeight[] GetSmoothedBoneWeights(int lerpIndex)
        {
            Debug.Assert(lerpIndex >= 0);

            while (lerpIndex >= m_SmoothedBoneWeights.Count && lerpIndex <= k_MaxSmoothIterations)
            {
                SmoothingUtility.SmoothWeights(m_SmoothedBoneWeights[^1], spriteMeshData.indices, boneCount, out var boneWeights);
                m_SmoothedBoneWeights.Add(boneWeights);
            }

            return m_SmoothedBoneWeights[Mathf.Min(lerpIndex, k_MaxSmoothIterations)];
        }

        static float GetLerpValue(float smoothValue)
        {
            Debug.Assert(smoothValue >= 0f);
            return smoothValue - Mathf.Floor(smoothValue);
        }

        static int GetLerpIndex(float smoothValue)
        {
            Debug.Assert(smoothValue >= 0f);
            return Mathf.RoundToInt(Mathf.Floor(smoothValue) + 1);
        }

        void StoreBoneWeights()
        {
            Debug.Assert(selection != null);

            m_StoredBoneWeights.Clear();

            for (var i = 0; i < spriteMeshData.vertexCount; i++)
            {
                var editableBoneWeight = spriteMeshData.vertexWeights[i];
                m_StoredBoneWeights.Add(editableBoneWeight.ToBoneWeight(false));
            }
        }

        void RestoreBoneWeights()
        {
            Debug.Assert(selection != null);

            for (var i = 0; i < spriteMeshData.vertexCount; i++)
            {
                var editableBoneWeight = spriteMeshData.vertexWeights[i];
                editableBoneWeight.SetFromBoneWeight(m_StoredBoneWeights[i]);
            }

            if (m_SmoothValues != null)
                Array.Clear(m_SmoothValues, 0, m_SmoothValues.Length);
        }
    }
}
