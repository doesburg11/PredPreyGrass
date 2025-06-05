#pragma warning disable 0168 // variable declared but not used.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine.Scripting;
using UnityEngine.U2D.Common;
using Unity.Collections;
using Unity.Profiling;
using UnityEngine.Rendering;
using UnityEngine.Scripting.APIUpdating;

namespace UnityEngine.U2D.Animation
{
    /// <summary>
    /// Represents vertex position.
    /// </summary>
    internal struct PositionVertex
    {
        /// <summary>
        /// Vertex position.
        /// </summary>
        public Vector3 position;
    }

    /// <summary>
    /// Represents vertex position and tangent.
    /// </summary>
    internal struct PositionTangentVertex
    {
        /// <summary>
        /// Vertex position.
        /// </summary>
        public Vector3 position;

        /// <summary>
        /// Vertex tangent.
        /// </summary>
        public Vector4 tangent;
    }

    /// <summary>
    /// The state of the Sprite Skin.
    /// </summary>
    public enum SpriteSkinState
    {
        /// <summary>
        /// Sprite Renderer doesn't contain a sprite.
        /// </summary>
        SpriteNotFound,

        /// <summary>
        /// Sprite referenced in the Sprite Renderer doesn't have skinning information.
        /// </summary>
        SpriteHasNoSkinningInformation,

        /// <summary>
        /// Sprite referenced in the Sprite Renderer doesn't have weights.
        /// </summary>
        SpriteHasNoWeights,

        /// <summary>
        /// Root transform is not assigned.
        /// </summary>
        RootTransformNotFound,

        /// <summary>
        /// Bone transform array is not assigned.
        /// </summary>
        InvalidTransformArray,

        /// <summary>
        /// Bone transform array has incorrect length.
        /// </summary>
        InvalidTransformArrayLength,

        /// <summary>
        /// One or more bone transforms is not assigned.
        /// </summary>
        TransformArrayContainsNull,

        /// <summary>
        /// Bone weights are invalid.
        /// </summary>
        InvalidBoneWeights,

        /// <summary>
        /// Sprite Skin is ready for deformation.
        /// </summary>
        Ready
    }

    /// <summary>
    /// Deforms the Sprite that is currently assigned to the SpriteRenderer in the same GameObject.
    /// </summary>
    [Preserve]
    [ExecuteInEditMode]
    [DefaultExecutionOrder(UpdateOrder.spriteSkinUpdateOrder)]
    [DisallowMultipleComponent]
    [RequireComponent(typeof(SpriteRenderer))]
    [AddComponentMenu("2D Animation/Sprite Skin")]
    [IconAttribute(IconUtility.IconPath + "Animation.SpriteSkin.png")]
    [MovedFrom("UnityEngine.U2D.Experimental.Animation")]
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.2d.animation@latest/index.html?subfolder=/manual/SpriteSkin.html")]
    public sealed class SpriteSkin : MonoBehaviour, IPreviewable, ISerializationCallbackReceiver
    {
        internal static class Profiling
        {
            public static readonly ProfilerMarker cacheCurrentSprite = new ProfilerMarker("SpriteSkin.CacheCurrentSprite");
            public static readonly ProfilerMarker cacheHierarchy = new ProfilerMarker("SpriteSkin.CacheHierarchy");
            public static readonly ProfilerMarker getSpriteBonesTransformFromGuid = new ProfilerMarker("SpriteSkin.GetSpriteBoneTransformsFromGuid");
            public static readonly ProfilerMarker getSpriteBonesTransformFromPath = new ProfilerMarker("SpriteSkin.GetSpriteBoneTransformsFromPath");
        }

        internal struct TransformData
        {
            public string fullName;
            public Transform transform;
        }

        [SerializeField]
        Transform m_RootBone;
        [SerializeField]
        Transform[] m_BoneTransforms = Array.Empty<Transform>();
        [SerializeField]
        Bounds m_Bounds;
        [SerializeField]
        bool m_AlwaysUpdate = true;
        [SerializeField]
        bool m_AutoRebind = false;

        // The deformed m_SpriteVertices stores all 'HOT' channels only in single-stream and essentially depends on Sprite Asset data.
        // The order of storage if present is POSITION, NORMALS, TANGENTS.
        NativeByteArray m_DeformedVertices;
        int m_CurrentDeformVerticesLength = 0;
        SpriteRenderer m_SpriteRenderer;
        int m_CurrentDeformSprite = 0;
        int m_SpriteId = 0;
        bool m_IsValid = false;
        SpriteSkinState m_State;
        int m_TransformsHash = 0;
        bool m_ForceCpuDeformation = false;

        int m_TextureId;
        int m_TransformId;
        NativeArray<int> m_BoneTransformId;
        int m_RootBoneTransformId;
        NativeCustomSlice<Vector3> m_SpriteVertices;
        NativeCustomSlice<Vector4> m_SpriteTangents;
        NativeCustomSlice<BoneWeight> m_SpriteBoneWeights;
        NativeCustomSlice<Matrix4x4> m_SpriteBindPoses;
        NativeCustomSlice<int> m_BoneTransformIdNativeSlice;
        bool m_SpriteHasTangents;
        int m_SpriteVertexStreamSize;
        int m_SpriteVertexCount;
        int m_SpriteTangentVertexOffset;
        int m_DataIndex = -1;
        bool m_BoneCacheUpdateToDate = false;

        internal Dictionary<int, List<TransformData>> hierarchyCache = new Dictionary<int, List<TransformData>>();

        NativeArray<int> m_OutlineIndexCache;
        NativeArray<Vector3> m_StaticOutlineVertexCache;
        NativeArray<Vector3> m_DeformedOutlineVertexCache;
        int m_VertexDeformationHash = 0;
        Sprite m_Sprite;

        internal NativeArray<int> boneTransformId => m_BoneTransformId;
        internal int rootBoneTransformId => m_RootBoneTransformId;
        internal DeformationMethods currentDeformationMethod { get; private set; }
        internal BaseDeformationSystem deformationSystem { get; private set; }

#if ENABLE_URP
        /// <summary>
        /// Returns an array of the outline indices.
        /// The indices are sorted and laid out with line topology.
        /// </summary>
        internal NativeArray<int> outlineIndices => m_OutlineIndexCache;

        /// <summary>
        /// Returns an array of the deformed outline vertices.
        /// </summary>
        internal NativeArray<Vector3> outlineVertices => m_DeformedOutlineVertexCache;
#endif

        /// <summary>
        /// Returns a hash which is updated every time the mesh is deformed.
        /// </summary>
        internal int vertexDeformationHash => m_VertexDeformationHash;

        internal Sprite sprite => m_Sprite;
        internal SpriteRenderer spriteRenderer => m_SpriteRenderer;
        internal NativeCustomSlice<BoneWeight> spriteBoneWeights => m_SpriteBoneWeights;

        /// <summary>
        /// Gets the index of the SpriteSkin in the SpriteSkinComposite.
        /// </summary>
        internal int dataIndex => m_DataIndex;

        internal void SetDataIndex(int index)
        {
            m_DataIndex = index;
        }

        /// <summary>
        /// Get and set the Auto Rebind property.
        /// When enabled, Sprite Skin attempts to automatically locate the Transform that is needed for the current Sprite assigned to the Sprite Renderer.
        /// </summary>
        public bool autoRebind
        {
            get => m_AutoRebind;
            set
            {
                if (m_AutoRebind == value)
                    return;

                m_AutoRebind = value;
                if (isActiveAndEnabled)
                {
                    CacheHierarchy();

                    m_CurrentDeformSprite = 0;
                    CacheCurrentSprite(m_AutoRebind);
                }
                else
                {
                    hierarchyCache.Clear();
                    CacheValidFlag();
                }
            }
        }

        /// <summary>
        /// Returns the Transform Components that are used for deformation.
        /// Do not modify elements of the returned array.
        /// </summary>
        public Transform[] boneTransforms => m_BoneTransforms;

        /// <summary>
        /// Sets the Transform Components that are used for deformation.
        /// </summary>
        /// <param name="boneTransformsArray">Array of new bone Transforms.</param>
        /// <returns>The state of the Sprite Skin.</returns>
        public SpriteSkinState SetBoneTransforms(Transform[] boneTransformsArray)
        {
            m_BoneTransforms = boneTransformsArray;

            if (isActiveAndEnabled)
                OnBoneTransformChanged();
            else
                CacheValidFlag();

            return m_State;
        }

        /// <summary>
        /// Returns the Transform Component that represents the root bone for deformation.
        /// </summary>
        public Transform rootBone => m_RootBone;

        /// <summary>
        /// Sets the Transform Component that represents the root bone for deformation.
        /// </summary>
        /// <param name="rootBoneTransform">Root bone Transform Component.</param>
        /// <returns>The state of the Sprite Skin.</returns>
        public SpriteSkinState SetRootBone(Transform rootBoneTransform)
        {
            m_RootBone = rootBoneTransform;

            if (isActiveAndEnabled)
            {
                CacheHierarchy();
                OnBoneTransformChanged();
            }
            else
            {
                hierarchyCache.Clear();
                CacheValidFlag();
            }

            return m_State;
        }

        internal Bounds bounds
        {
            get => m_Bounds;
            set => m_Bounds = value;
        }

        /// <summary>
        /// Determines if the SpriteSkin executes even if the associated
        /// SpriteRenderer has been culled from view.
        /// </summary>
        public bool alwaysUpdate
        {
            get => m_AlwaysUpdate;
            set => m_AlwaysUpdate = value;
        }

        /// <summary>
        /// Always run a deformation pass on the CPU.<br/>
        /// If GPU deformation is enabled, enabling forceCpuDeformation will cause the deformation to run twice, one time on the CPU and one time on the GPU.
        /// </summary>
        public bool forceCpuDeformation
        {
            get => m_ForceCpuDeformation;
            set
            {
                if (m_ForceCpuDeformation == value)
                    return;
                m_ForceCpuDeformation = value;

                if (isActiveAndEnabled)
                {
                    UpdateSpriteDeformationData();
                    deformationSystem?.CopyToSpriteSkinData(this);
                }
            }
        }

        /// <summary>
        /// Resets the bone transforms to the bind pose.
        /// </summary>
        /// <returns>True if successful.</returns>
        public bool ResetBindPose()
        {
            if (!isValid)
                return false;

            var spriteBones = spriteRenderer.sprite.GetBones();
            for (var i = 0; i < boneTransforms.Length; ++i)
            {
                var boneTransform = boneTransforms[i];
                var spriteBone = spriteBones[i];

                if (spriteBone.parentId != -1)
                {
                    boneTransform.localPosition = spriteBone.position;
                    boneTransform.localRotation = spriteBone.rotation;
                    boneTransform.localScale = Vector3.one;
                }
            }

            return true;
        }

        internal bool isValid => this.Validate() == SpriteSkinState.Ready;

#if UNITY_EDITOR
        internal static Events.UnityEvent onDrawGizmos = new Events.UnityEvent();

        void OnDrawGizmos() => onDrawGizmos.Invoke();

        internal bool ignoreNextSpriteChange { get; set; } = true;
#endif

        internal void Awake()
        {
            m_SpriteRenderer = GetComponent<SpriteRenderer>();
            m_Sprite = m_SpriteRenderer.sprite;
            m_SpriteId = m_Sprite != null ? m_Sprite.GetInstanceID() : 0;
        }

        void OnEnable()
        {
            m_TransformId = gameObject.transform.GetInstanceID();
            m_TransformsHash = 0;
            currentDeformationMethod = SpriteSkinUtility.CanSpriteSkinUseGpuDeformation(this) ? DeformationMethods.Gpu : DeformationMethods.Cpu;

            Awake();

            CacheCurrentSprite(m_AutoRebind);
            UpdateSpriteDeformationData();

            if (hierarchyCache.Count == 0)
                CacheHierarchy();

            RefreshBoneTransforms();

            DeformationManager.instance.AddSpriteSkin(this);
            SpriteSkinContainer.instance.AddSpriteSkin(this);

            m_SpriteRenderer.RegisterSpriteChangeCallback(OnSpriteChanged);
        }

        void OnDisable()
        {
            m_SpriteRenderer.UnregisterSpriteChangeCallback(OnSpriteChanged);

            DeactivateSkinning();
            BufferManager.instance.ReturnBuffer(GetInstanceID());
            deformationSystem?.RemoveSpriteSkin(this);
            deformationSystem = null;
            SpriteSkinContainer.instance.RemoveSpriteSkin(this);
            ResetBoneTransformIdCache();
            DisposeOutlineCaches();
        }

        void RefreshBoneTransforms()
        {
            DeformationManager.instance.RemoveBoneTransforms(this);
            CacheBoneTransformIds();
            DeformationManager.instance.AddSpriteSkinBoneTransform(this);

            CacheValidFlag();
        }

        void OnSpriteChanged(SpriteRenderer updatedSpriteRenderer)
        {
            m_Sprite = updatedSpriteRenderer.sprite;
            m_SpriteId = m_Sprite != null ? m_Sprite.GetInstanceID() : 0;
        }

        void CacheBoneTransformIds()
        {
            m_BoneCacheUpdateToDate = true;

            var boneCount = 0;
            for (var i = 0; i < boneTransforms?.Length; ++i)
            {
                if (boneTransforms[i] != null)
                    ++boneCount;
            }

            if (m_BoneTransformId != default && m_BoneTransformId.IsCreated)
                NativeArrayHelpers.ResizeIfNeeded(ref m_BoneTransformId, boneCount);
            else
                m_BoneTransformId = new NativeArray<int>(boneCount, Allocator.Persistent);

            m_RootBoneTransformId = rootBone != null ? rootBone.GetInstanceID() : 0;
            m_BoneTransformIdNativeSlice = new NativeCustomSlice<int>(m_BoneTransformId);
            for (int i = 0, j = 0; i < boneTransforms?.Length; ++i)
            {
                if (boneTransforms[i] != null)
                {
                    m_BoneTransformId[j] = boneTransforms[i].GetInstanceID();
                    ++j;
                }
            }
        }

        void OnBoneTransformChanged()
        {
            RefreshBoneTransforms();
            deformationSystem?.CopyToSpriteSkinData(this);
            SpriteSkinContainer.instance.BoneTransformsChanged(this);
        }

        /// <summary>
        /// Called before object is serialized.
        /// </summary>
        public void OnBeforeSerialize()
        {
            OnBeforeSerializeBatch();
        }

        /// <summary>
        /// Called after object is deserialized.
        /// </summary>
        public void OnAfterDeserialize()
        {
            OnAfterSerializeBatch();
        }

        void OnBeforeSerializeBatch() { }

        void OnAfterSerializeBatch()
        {
#if UNITY_EDITOR
            m_BoneCacheUpdateToDate = false;
#endif
        }

        internal void OnEditorEnable()
        {
            Awake();
        }

        SpriteSkinState CacheValidFlag()
        {
            m_State = this.Validate();
            m_IsValid = m_State == SpriteSkinState.Ready;
            if (!m_IsValid)
                DeactivateSkinning();

            return m_State;
        }

        internal bool BatchValidate()
        {
            if (!m_BoneCacheUpdateToDate)
                RefreshBoneTransforms();

            CacheCurrentSprite(m_AutoRebind);
            var hasSprite = m_CurrentDeformSprite != 0;
            return m_IsValid && hasSprite && m_SpriteRenderer.enabled && (alwaysUpdate || m_SpriteRenderer.isVisible);
        }

        void Reset()
        {
            Awake();
            if (isActiveAndEnabled)
            {
                CacheValidFlag();

                if (!m_BoneCacheUpdateToDate)
                    RefreshBoneTransforms();

                deformationSystem?.CopyToSpriteSkinData(this);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        void ResetBoneTransformIdCache()
        {
            m_BoneTransformId.DisposeIfCreated();
            m_BoneTransformId = default;

            m_RootBoneTransformId = -1;
            m_BoneCacheUpdateToDate = false;
        }

        internal NativeByteArray GetDeformedVertices(int spriteVertexCount)
        {
            if (sprite != null)
            {
                if (m_CurrentDeformVerticesLength != spriteVertexCount)
                {
                    m_TransformsHash = 0;
                    m_CurrentDeformVerticesLength = spriteVertexCount;
                }
            }
            else
            {
                m_CurrentDeformVerticesLength = 0;
            }

            m_DeformedVertices = BufferManager.instance.GetBuffer(GetInstanceID(), m_CurrentDeformVerticesLength);
            return m_DeformedVertices;
        }

        /// <summary>
        /// Returns whether this SpriteSkin has currently deformed vertices.
        /// </summary>
        /// <returns>Returns true if this SpriteSkin has currently deformed vertices. Returns false otherwise.</returns>
        public bool HasCurrentDeformedVertices()
        {
            if (!m_IsValid)
                return false;

            return m_DataIndex >= 0 && deformationSystem != null && deformationSystem.IsSpriteSkinActiveForDeformation(this);
        }

        /// <summary>
        /// Gets a byte array to the currently deformed vertices for this SpriteSkin.
        /// </summary>
        /// <returns>Returns a reference to the currently deformed vertices. This is valid only for this calling frame.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when there are no currently deformed vertices.
        /// HasCurrentDeformedVertices can be used to verify if there are any deformed vertices available.
        /// </exception>
        internal NativeArray<byte> GetCurrentDeformedVertices()
        {
            if (!m_IsValid)
                throw new InvalidOperationException("The SpriteSkin deformation is not valid.");
            if (m_DataIndex < 0)
                throw new InvalidOperationException("There are no currently deformed vertices.");

            var buffer = deformationSystem?.GetDeformableBufferForSpriteSkin(this) ?? default;
            if (buffer == default)
                throw new InvalidOperationException("There are no currently deformed vertices.");

            return buffer;
        }

        /// <summary>
        /// Gets an array of currently deformed position vertices for this SpriteSkin.
        /// </summary>
        /// <returns>Returns a reference to the currently deformed vertices. This is valid only for this calling frame.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when there are no currently deformed vertices or if the deformed vertices does not contain only
        /// position data. HasCurrentDeformedVertices can be used to verify if there are any deformed vertices available.
        /// </exception>
        internal NativeSlice<PositionVertex> GetCurrentDeformedVertexPositions()
        {
            if (!m_IsValid)
                throw new InvalidOperationException("The SpriteSkin deformation is not valid.");

            if (sprite.HasVertexAttribute(VertexAttribute.Tangent))
                throw new InvalidOperationException("This SpriteSkin has deformed tangents");
            if (!sprite.HasVertexAttribute(VertexAttribute.Position))
                throw new InvalidOperationException("This SpriteSkin does not have deformed positions.");

            var deformedBuffer = GetCurrentDeformedVertices();
            return deformedBuffer.Slice().SliceConvert<PositionVertex>();
        }

        /// <summary>
        /// Gets an array of currently deformed position and tangent vertices for this SpriteSkin.
        /// </summary>
        /// <returns>
        /// Returns a reference to the currently deformed position and tangent vertices. This is valid only for this calling frame.
        /// </returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when there are no currently deformed vertices or if the deformed vertices does not contain only
        /// position and tangent data. HasCurrentDeformedVertices can be used to verify if there are any deformed vertices available.
        /// </exception>
        internal NativeSlice<PositionTangentVertex> GetCurrentDeformedVertexPositionsAndTangents()
        {
            if (!m_IsValid)
                throw new InvalidOperationException("The SpriteSkin deformation is not valid.");

            if (!sprite.HasVertexAttribute(VertexAttribute.Tangent))
                throw new InvalidOperationException("This SpriteSkin does not have deformed tangents");
            if (!sprite.HasVertexAttribute(VertexAttribute.Position))
                throw new InvalidOperationException("This SpriteSkin does not have deformed positions.");

            var deformedBuffer = GetCurrentDeformedVertices();
            return deformedBuffer.Slice().SliceConvert<PositionTangentVertex>();
        }

        /// <summary>
        /// Gets an enumerable to iterate through all deformed vertex positions of this SpriteSkin.
        /// </summary>
        /// <returns>Returns an IEnumerable to deformed vertex positions.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when there is no vertex positions or deformed vertices.
        /// HasCurrentDeformedVertices can be used to verify if there are any deformed vertices available.
        /// </exception>
        public IEnumerable<Vector3> GetDeformedVertexPositionData()
        {
            if (!m_IsValid)
                throw new InvalidOperationException("The SpriteSkin deformation is not valid.");

            var hasPosition = sprite.HasVertexAttribute(VertexAttribute.Position);
            if (!hasPosition)
                throw new InvalidOperationException("Sprite does not have vertex position data.");

            var rawBuffer = GetCurrentDeformedVertices();
            var rawSlice = rawBuffer.Slice(sprite.GetVertexStreamOffset(VertexAttribute.Position));
            return new NativeCustomSliceEnumerator<Vector3>(rawSlice, m_SpriteVertexCount, m_SpriteVertexStreamSize);
        }

        /// <summary>
        /// Gets an enumerable to iterate through all deformed vertex tangents of this SpriteSkin.
        /// </summary>
        /// <returns>Returns an IEnumerable to deformed vertex tangents.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when there is no vertex tangents or deformed vertices.
        /// HasCurrentDeformedVertices can be used to verify if there are any deformed vertices available.
        /// </exception>
        public IEnumerable<Vector4> GetDeformedVertexTangentData()
        {
            if (!m_IsValid)
                throw new InvalidOperationException("The SpriteSkin deformation is not valid.");

            var hasTangent = sprite.HasVertexAttribute(VertexAttribute.Tangent);
            if (!hasTangent)
                throw new InvalidOperationException("Sprite does not have vertex tangent data.");

            var rawBuffer = GetCurrentDeformedVertices();
            var rawSlice = rawBuffer.Slice(sprite.GetVertexStreamOffset(VertexAttribute.Tangent));
            return new NativeCustomSliceEnumerator<Vector4>(rawSlice, m_SpriteVertexCount, m_SpriteVertexStreamSize);
        }

        void DisposeOutlineCaches()
        {
            m_OutlineIndexCache.DisposeIfCreated();
            m_StaticOutlineVertexCache.DisposeIfCreated();
            m_DeformedOutlineVertexCache.DisposeIfCreated();

            m_OutlineIndexCache = default;
            m_StaticOutlineVertexCache = default;
            m_DeformedOutlineVertexCache = default;
        }

        /// <summary>
        /// Used by the animation clip preview window.
        /// Recommended to not use outside of this purpose.
        /// </summary>
        public void OnPreviewUpdate()
        {
#if UNITY_EDITOR
            if (IsInGUIUpdateLoop())
                Deform();
#endif
        }

        static bool IsInGUIUpdateLoop() => Event.current != null;

        void Deform()
        {
            if (m_SpriteRenderer.sprite != m_Sprite)
                OnSpriteChanged(m_SpriteRenderer);

            CacheCurrentSprite(m_AutoRebind);
            if (isValid && enabled && (alwaysUpdate || m_SpriteRenderer.isVisible))
            {
                var transformHash = SpriteSkinUtility.CalculateTransformHash(this);
                var spriteVertexCount = sprite.GetVertexStreamSize() * sprite.GetVertexCount();
                if (spriteVertexCount > 0 && (m_TransformsHash != transformHash || vertexDeformationHash != GetNewVertexDeformationHash()))
                {
                    var inputVertices = GetDeformedVertices(spriteVertexCount);
                    SpriteSkinUtility.Deform(sprite, gameObject.transform.worldToLocalMatrix, boneTransforms, inputVertices.array);
                    SpriteSkinUtility.UpdateBounds(this, inputVertices.array);
                    InternalEngineBridge.SetDeformableBuffer(spriteRenderer, inputVertices.array);
                    m_TransformsHash = transformHash;
                    m_CurrentDeformSprite = m_SpriteId;

                    PostDeform(true);
                }
            }
            else if (!InternalEngineBridge.IsUsingDeformableBuffer(spriteRenderer, IntPtr.Zero))
            {
                DeactivateSkinning();
            }
        }

        internal void PostDeform(bool didDeform)
        {
            if (didDeform)
            {
#if ENABLE_URP
                UpdateDeformedOutlineCache();
#endif
                m_VertexDeformationHash = GetNewVertexDeformationHash();
            }
        }

        void CacheCurrentSprite(bool rebind)
        {
            if (m_CurrentDeformSprite == m_SpriteId)
                return;

            using (Profiling.cacheCurrentSprite.Auto())
            {
                DeactivateSkinning();
                m_CurrentDeformSprite = m_SpriteId;
                if (rebind && m_CurrentDeformSprite != 0 && rootBone != null)
                {
                    if (!SpriteSkinHelpers.GetSpriteBonesTransforms(this, out var transforms))
                        Debug.LogWarning($"Rebind failed for {name}. Could not find all bones required by the Sprite: {sprite.name}.");
                    SetBoneTransforms(transforms);
                }

                UpdateSpriteDeformationData();
                deformationSystem?.CopyToSpriteSkinData(this);

                CacheValidFlag();
                m_TransformsHash = 0;
            }
        }

        void UpdateSpriteDeformationData()
        {
#if ENABLE_URP
            CacheSpriteOutline();
#endif

            if (sprite == null)
            {
                m_TextureId = 0;
                m_SpriteVertices = NativeCustomSlice<Vector3>.Default();
                m_SpriteTangents = NativeCustomSlice<Vector4>.Default();
                m_SpriteBoneWeights = NativeCustomSlice<BoneWeight>.Default();
                m_SpriteBindPoses = NativeCustomSlice<Matrix4x4>.Default();
                m_SpriteHasTangents = false;
                m_SpriteVertexStreamSize = 0;
                m_SpriteVertexCount = 0;
                m_SpriteTangentVertexOffset = 0;
            }
            else
            {
                m_TextureId = sprite.texture != null ? sprite.texture.GetInstanceID() : 0;
                var cacheFullMesh = currentDeformationMethod == DeformationMethods.Cpu || forceCpuDeformation;
                if (cacheFullMesh)
                {
                    m_SpriteVertices = new NativeCustomSlice<Vector3>(sprite.GetVertexAttribute<Vector3>(VertexAttribute.Position));
                    m_SpriteVertexCount = sprite.GetVertexCount();
                    m_SpriteVertexStreamSize = sprite.GetVertexStreamSize();

                    m_SpriteTangents = new NativeCustomSlice<Vector4>(sprite.GetVertexAttribute<Vector4>(VertexAttribute.Tangent));
                    m_SpriteHasTangents = sprite.HasVertexAttribute(VertexAttribute.Tangent);
                    m_SpriteTangentVertexOffset = sprite.GetVertexStreamOffset(VertexAttribute.Tangent);
                }
                else
                {
                    m_SpriteVertices = new NativeCustomSlice<Vector3>(m_StaticOutlineVertexCache);
                    m_SpriteVertexCount = m_SpriteVertices.length;
                    m_SpriteVertexStreamSize = sizeof(float) * 3;

                    m_SpriteTangents = new NativeCustomSlice<Vector4>(sprite.GetVertexAttribute<Vector4>(VertexAttribute.Tangent));
                    m_SpriteHasTangents = false;
                    m_SpriteTangentVertexOffset = 0;
                }

                m_SpriteBoneWeights = new NativeCustomSlice<BoneWeight>(sprite.GetVertexAttribute<BoneWeight>(VertexAttribute.BlendWeight));
                m_SpriteBindPoses = new NativeCustomSlice<Matrix4x4>(sprite.GetBindPoses());
            }
        }

#if ENABLE_URP
        void UpdateDeformedOutlineCache()
        {
            if (sprite == null)
                return;
            if (!m_OutlineIndexCache.IsCreated || !m_DeformedOutlineVertexCache.IsCreated)
                return;
            if (!HasCurrentDeformedVertices())
                return;

            var buffer = GetCurrentDeformedVertices();
            var indexCache = m_OutlineIndexCache;
            var vertexCache = m_DeformedOutlineVertexCache;
            BurstedSpriteSkinUtilities.SetVertexPositionFromByteBuffer(in buffer, in indexCache, ref vertexCache, m_SpriteVertexStreamSize);
            m_DeformedOutlineVertexCache = vertexCache;
        }

        void CacheSpriteOutline()
        {
            DisposeOutlineCaches();

            if (sprite == null)
                return;

            CacheOutlineIndices(out var maxIndex);
            var cacheSize = maxIndex + 1;
            CacheOutlineVertices(cacheSize);
        }

        void CacheOutlineIndices(out int maxIndex)
        {
            var indices = sprite.GetIndices();
            var edgeNativeArr = MeshUtilities.GetOutlineEdges(in indices);

            m_OutlineIndexCache = new NativeArray<int>(edgeNativeArr.Length * 2, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            maxIndex = 0;
            for (var i = 0; i < edgeNativeArr.Length; ++i)
            {
                var indexX = edgeNativeArr[i].x;
                var indexY = edgeNativeArr[i].y;
                m_OutlineIndexCache[i * 2] = indexX;
                m_OutlineIndexCache[(i * 2) + 1] = indexY;

                if (indexX > maxIndex)
                    maxIndex = indexX;
                if (indexY > maxIndex)
                    maxIndex = indexY;
            }

            edgeNativeArr.Dispose();
        }

        void CacheOutlineVertices(int cacheSize)
        {
            m_DeformedOutlineVertexCache = new NativeArray<Vector3>(cacheSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            m_StaticOutlineVertexCache = new NativeArray<Vector3>(cacheSize, Allocator.Persistent);

            var vertices = sprite.GetVertexAttribute<Vector3>(VertexAttribute.Position);
            var vertexCache = m_StaticOutlineVertexCache;
            for (var i = 0; i < m_OutlineIndexCache.Length; ++i)
            {
                var index = m_OutlineIndexCache[i];
                vertexCache[index] = vertices[index];
            }

            m_StaticOutlineVertexCache = vertexCache;
        }
#endif
        internal void CopyToSpriteSkinData(ref SpriteSkinData data)
        {
            if (!m_BoneCacheUpdateToDate)
                RefreshBoneTransforms();

            CacheCurrentSprite(m_AutoRebind);

            data.vertices = m_SpriteVertices;
            data.boneWeights = m_SpriteBoneWeights;
            data.bindPoses = m_SpriteBindPoses;
            data.tangents = m_SpriteTangents;
            data.hasTangents = m_SpriteHasTangents;
            data.spriteVertexStreamSize = m_SpriteVertexStreamSize;
            data.spriteVertexCount = m_SpriteVertexCount;
            data.tangentVertexOffset = m_SpriteTangentVertexOffset;
            data.transformId = m_TransformId;
            data.boneTransformId = m_BoneTransformIdNativeSlice;
        }

        internal bool NeedToUpdateDeformationCache()
        {
            var newTextureId = sprite.texture != null ? sprite.texture.GetInstanceID() : 0;
            var needUpdate = newTextureId != m_TextureId;
            if (needUpdate)
            {
                UpdateSpriteDeformationData();
                deformationSystem?.CopyToSpriteSkinData(this);
            }

            return needUpdate;
        }

        internal void CacheHierarchy()
        {
            using (Profiling.cacheHierarchy.Auto())
            {
                hierarchyCache.Clear();
                if (rootBone == null || !m_AutoRebind)
                    return;

                var boneCount = CountChildren(rootBone);
                hierarchyCache.EnsureCapacity(boneCount + 1);
                SpriteSkinHelpers.CacheChildren(rootBone, hierarchyCache);

                foreach (var entry in hierarchyCache)
                {
                    if (entry.Value.Count == 1)
                        continue;
                    var count = entry.Value.Count;
                    for (var i = 0; i < count; ++i)
                    {
                        var transformEntry = entry.Value[i];
                        transformEntry.fullName = SpriteSkinHelpers.GenerateTransformPath(rootBone, transformEntry.transform);
                        entry.Value[i] = transformEntry;
                    }
                }
            }
        }

        internal void DeactivateSkinning()
        {
            if (m_SpriteRenderer != null)
            {
                var currentSprite = sprite;
                if (currentSprite != null)
                    InternalEngineBridge.SetLocalAABB(m_SpriteRenderer, currentSprite.bounds);

                m_SpriteRenderer.DeactivateDeformableBuffer();
            }

            m_TransformsHash = 0;
        }

        internal void ResetSprite()
        {
            m_CurrentDeformSprite = 0;
            CacheValidFlag();
        }

        internal void SetDeformationSystem(BaseDeformationSystem newDeformationSystem)
        {
            deformationSystem = newDeformationSystem;
            currentDeformationMethod = deformationSystem.deformationMethod;
        }

        static int CountChildren(Transform transform)
        {
            var childCount = transform.childCount;
            var count = childCount;
            for (var i = 0; i < childCount; ++i)
                count += CountChildren(transform.GetChild(i));

            return count;
        }

        static int GetNewVertexDeformationHash() => Time.frameCount;
    }
}
