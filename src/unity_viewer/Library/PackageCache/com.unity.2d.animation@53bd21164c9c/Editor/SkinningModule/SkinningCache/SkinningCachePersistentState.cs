using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal interface ISkinningCachePersistentState
    {
        string lastSpriteId
        {
            get;
            set;
        }

        Tools lastUsedTool
        {
            get;
            set;
        }

        List<int> lastBoneSelectionIds
        {
            get;
        }

        Texture2D lastTexture
        {
            get;
            set;
        }

        SerializableDictionary<int, BonePose> lastPreviewPose
        {
            get;
        }

        SerializableDictionary<int, bool> lastBoneVisibility
        {
            get;
        }

        SerializableDictionary<int, bool> lastBoneExpansion
        {
            get;
        }

        SerializableDictionary<string, bool> lastSpriteVisibility
        {
            get;
        }

        SerializableDictionary<int, bool> lastGroupVisibility
        {
            get;
        }

        SkinningMode lastMode
        {
            get;
            set;
        }

        bool lastVisibilityToolActive
        {
            get;
            set;
        }

        int lastVisibilityToolIndex
        {
            get;
            set;
        }

        IndexedSelection lastVertexSelection
        {
            get;
        }

        float lastBrushSize
        {
            get;
            set;
        }

        float lastBrushHardness
        {
            get;
            set;
        }

        float lastBrushStep
        {
            get;
            set;
        }
    }

    [Serializable]
    internal class SkinningCachePersistentState
        : ScriptableSingleton<SkinningCachePersistentState>
        , ISkinningCachePersistentState
    {
        [SerializeField] Tools m_LastUsedTool = Tools.EditPose;

        [SerializeField] SkinningMode m_LastMode = SkinningMode.Character;

        [SerializeField] string m_LastSpriteId = String.Empty;

        [SerializeField] List<int> m_LastBoneSelectionIds = new List<int>();

        [SerializeField] Texture2D m_LastTexture;

        [SerializeField]
        SerializableDictionary<int, BonePose> m_SkeletonPreviewPose =
            new SerializableDictionary<int, BonePose>();

        [SerializeField]
        SerializableDictionary<int, bool> m_BoneVisibility =
            new SerializableDictionary<int, bool>();

        [SerializeField]
        SerializableDictionary<int, bool> m_BoneExpansion =
            new SerializableDictionary<int, bool>();

        [SerializeField]
        SerializableDictionary<string, bool> m_SpriteVisibility =
            new SerializableDictionary<string, bool>();

        [SerializeField]
        SerializableDictionary<int, bool> m_GroupVisibility =
            new SerializableDictionary<int, bool>();

        [SerializeField] IndexedSelection m_VertexSelection;

        [SerializeField] bool m_VisibilityToolActive;
        [SerializeField] int m_VisibilityToolIndex = -1;

        [SerializeField] float m_LastBrushSize = 25f;
        [SerializeField] float m_LastBrushHardness = 1f;
        [SerializeField] float m_LastBrushStep = 20f;

        public SkinningCachePersistentState()
        {
            m_VertexSelection = new IndexedSelection();
        }

        void OnEnable()
        {
            name = GetType().ToString();
        }

        void OnDisable()
        {
            Undo.ClearUndo(this);
        }

        public void SetDefault()
        {
            m_LastUsedTool = Tools.EditPose;
            m_LastMode = SkinningMode.Character;
            m_LastSpriteId = String.Empty;
            m_LastBoneSelectionIds.Clear();
            m_LastTexture = null;
            m_VertexSelection.Clear();
            m_SkeletonPreviewPose.Clear();
            m_BoneVisibility.Clear();
            m_BoneExpansion.Clear();
            m_SpriteVisibility.Clear();
            m_GroupVisibility.Clear();
            m_VisibilityToolActive = false;
            m_VisibilityToolIndex = -1;
        }

        public string lastSpriteId
        {
            get => m_LastSpriteId;
            set => m_LastSpriteId = value;
        }

        public Tools lastUsedTool
        {
            get => m_LastUsedTool;
            set => m_LastUsedTool = value;
        }

        public List<int> lastBoneSelectionIds => m_LastBoneSelectionIds;

        public Texture2D lastTexture
        {
            get => m_LastTexture;
            set
            {
                if (value != m_LastTexture)
                {
                    m_LastMode = SkinningMode.Character;
                    m_LastSpriteId = string.Empty;
                    m_LastBoneSelectionIds.Clear();
                    m_VertexSelection.Clear();
                    m_SkeletonPreviewPose.Clear();
                    m_BoneVisibility.Clear();
                    m_BoneExpansion.Clear();
                    m_SpriteVisibility.Clear();
                    m_GroupVisibility.Clear();
                }

                m_LastTexture = value;
            }
        }

        public SerializableDictionary<int, BonePose> lastPreviewPose => m_SkeletonPreviewPose;

        public SerializableDictionary<int, bool> lastBoneVisibility => m_BoneVisibility;

        public SerializableDictionary<int, bool> lastBoneExpansion => m_BoneExpansion;

        public SerializableDictionary<string, bool> lastSpriteVisibility => m_SpriteVisibility;

        public SerializableDictionary<int, bool> lastGroupVisibility => m_GroupVisibility;

        public SkinningMode lastMode
        {
            get => m_LastMode;
            set => m_LastMode = value;
        }

        public bool lastVisibilityToolActive
        {
            get => m_VisibilityToolActive;
            set => m_VisibilityToolActive = value;
        }

        public int lastVisibilityToolIndex
        {
            get => m_VisibilityToolIndex;
            set => m_VisibilityToolIndex = value;
        }

        public IndexedSelection lastVertexSelection => m_VertexSelection;

        public float lastBrushSize
        {
            get => m_LastBrushSize;
            set => m_LastBrushSize = value;
        }

        public float lastBrushHardness
        {
            get => m_LastBrushHardness;
            set => m_LastBrushHardness = value;
        }

        public float lastBrushStep
        {
            get => m_LastBrushStep;
            set => m_LastBrushStep = value;
        }
    }
}