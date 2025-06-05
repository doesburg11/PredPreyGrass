using System.Collections.Generic;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;
#if UNITY_EDITOR
using UnityEditor.U2D;
#endif

namespace UnityEngine.U2D
{

    /// <summary>
    /// SpriteShapeController component contains Spline and SpriteShape Profile information that is used when generating SpriteShape geometry.
    /// </summary>
    [ExecuteInEditMode]
    [RequireComponent(typeof(SpriteShapeRenderer))]
    [DisallowMultipleComponent]
    [HelpURLAttribute("https://docs.unity3d.com/Packages/com.unity.2d.spriteshape@latest/index.html?subfolder=/manual/SSController.html")]
    public class SpriteShapeController : MonoBehaviour
    {
        // Internal Dataset.
        const float s_DistanceTolerance = 0.001f;

        // Cached Objects.
        SpriteShape m_ActiveSpriteShape;
        EdgeCollider2D m_EdgeCollider2D;
        PolygonCollider2D m_PolygonCollider2D;
        SpriteShapeRenderer m_SpriteShapeRenderer;
        SpriteShapeGeometryCache m_SpriteShapeGeometryCache;

        Sprite[] m_SpriteArray = new Sprite[0];
        Sprite[] m_EdgeSpriteArray = new Sprite[0];
        Sprite[] m_CornerSpriteArray = new Sprite[0];
        AngleRangeInfo[] m_AngleRangeInfoArray = new AngleRangeInfo[0];

        // Required for Generation.
        NativeArray<float2> m_ColliderData;
        NativeArray<float2> m_ShadowData;
        NativeArray<Vector4> m_TangentData;
        NativeArray<SpriteShapeGeneratorStats> m_Statistics;

        // Renderer Stuff.
        bool m_DynamicOcclusionLocal;
        bool m_DynamicOcclusionOverriden;
        bool m_TessellationNeedsFallback = false;

        // Hash Check.
        int m_ActiveSplineHash = 0;
        int m_ActiveSpriteShapeHash = 0;
        int m_MaxArrayCount = 0;
        JobHandle m_JobHandle;
        SpriteShapeParameters m_ActiveShapeParameters;

        // Serialized Data.
        [SerializeField]
        Spline m_Spline = new Spline();
        [SerializeField]
        SpriteShape m_SpriteShape;

        [SerializeField]
        float m_FillPixelPerUnit = 100.0f;
        [SerializeField]
        float m_StretchTiling = 1.0f;
        [SerializeField]
        int m_SplineDetail;
        [SerializeField]
        bool m_AdaptiveUV;
        [SerializeField]
        bool m_StretchUV;
        [SerializeField]
        bool m_WorldSpaceUV;
        [SerializeField]
        float m_CornerAngleThreshold = 30.0f;
        [SerializeField]
        int m_ColliderDetail;
        [SerializeField, Range(-0.5f, 0.5f)]
        float m_ColliderOffset;
        [SerializeField]
        bool m_UpdateCollider = true;
        [SerializeField]
        bool m_EnableTangents = false;
        [SerializeField]
        [HideInInspector]
        bool m_GeometryCached = false;
        [SerializeField]
        bool m_UTess2D = true;
        [SerializeField]
        bool m_UpdateShadow = false;
        [SerializeField]
        int m_ShadowDetail = (int)QualityDetail.High;
        [SerializeField, Range(-0.5f, 0.5f)]
        float m_ShadowOffset = 0.5f;
        [SerializeField]
        float m_BoundsScale = 2.0f;

        [SerializeField]
        SpriteShapeGeometryCreator m_Creator;
        [SerializeField]
        List<SpriteShapeGeometryModifier> m_Modifiers = new List<SpriteShapeGeometryModifier>();
        [SerializeField]
        List<Vector2> m_ColliderSegment = new List<Vector2>();
        [SerializeField]
        List<Vector2> m_ShadowSegment = new List<Vector2>();

        internal static readonly ProfilerMarker generateGeometry = new ProfilerMarker("SpriteShape.GenerateGeometry");
        internal static readonly ProfilerMarker generateCollider = new ProfilerMarker("SpriteShape.GenerateCollider");

#region GetSet

        internal int maxArrayCount
        {
            get { return m_MaxArrayCount; }
            set { m_MaxArrayCount = value; }
        }

        internal bool geometryCached
        {
            get { return m_GeometryCached; }
            set { m_GeometryCached = value; }
        }

        internal int splineHashCode
        {
            get { return m_ActiveSplineHash; }
        }

        internal Sprite[] spriteArray
        {
            get { return m_SpriteArray; }
        }

        internal SpriteShapeParameters spriteShapeParameters
        {
            get { return m_ActiveShapeParameters; }
        }

        internal SpriteShapeGeometryCache spriteShapeGeometryCache
        {
            get
            {
                if (!m_SpriteShapeGeometryCache)
                {
                    bool b = TryGetComponent(typeof(SpriteShapeGeometryCache), out Component comp);
                    m_SpriteShapeGeometryCache = b ? (comp as SpriteShapeGeometryCache) : null;
                }
                return m_SpriteShapeGeometryCache;
            }
        }

        internal Sprite[] cornerSpriteArray
        {
            get { return m_CornerSpriteArray; }
        }

        internal Sprite[] edgeSpriteArray
        {
            get { return m_EdgeSpriteArray; }
        }

        internal NativeArray<float2> shadowData
        {
            get { return m_ShadowData; }
        }

        /// <summary> Angle Ranges </summary>
        public AngleRangeInfo[] angleRangeInfoArray
        {
            get { return m_AngleRangeInfoArray; }
        }

        /// <summary>Get/Set SpriteShape Geometry Creator. </summary>
        public SpriteShapeGeometryCreator spriteShapeCreator
        {
            get
            {
                if (m_Creator == null)
                    m_Creator = SpriteShapeDefaultCreator.defaultInstance;
                return m_Creator;
            }
            set
            {
                if (value != null)
                    m_Creator = value;
            }
        }

        /// <summary>Get a list of Modifiers. </summary>
        public List<SpriteShapeGeometryModifier> modifiers
        {
            get { return m_Modifiers; }
        }

        /// <summary>Hash code for SpriteShape used to check for changes. </summary>
        public int spriteShapeHashCode
        {
            get { return m_ActiveSpriteShapeHash; }
        }

        /// <summary>Defines whether UV for fill geometry uses local or global space. </summary>
        public bool worldSpaceUVs
        {
            get { return m_WorldSpaceUV; }
            set { m_WorldSpaceUV = value; }
        }

        /// <summary>Defines pixel per unit for fill geometry UV generation. </summary>
        public float fillPixelsPerUnit
        {
            get { return m_FillPixelPerUnit; }
            set { m_FillPixelPerUnit = value; }
        }

        /// <summary>Enable tangent channel when generating SpriteShape geometry (used in Shaders) </summary>
        public bool enableTangents
        {
            get { return m_EnableTangents; }
            set { m_EnableTangents = value; }
        }

        /// <summary>Stretch tiling for inner fill geometry UV generation. </summary>
        public float stretchTiling
        {
            get { return m_StretchTiling; }
            set { m_StretchTiling = value; }
        }

        /// <summary>Level of detail for generated geometry. </summary>
        public int splineDetail
        {
            get { return m_SplineDetail; }
            set { m_SplineDetail = Mathf.Max(0, value); }
        }

        /// <summary>Level of detail for geometry generated for colliders. </summary>
        public int colliderDetail
        {
            get { return m_ColliderDetail; }
            set { m_ColliderDetail = Mathf.Max(0, value); }
        }

        /// <summary>Offset for colliders. </summary>
        public float colliderOffset
        {
            get { return m_ColliderOffset; }
            set { m_ColliderOffset = value; }
        }

        /// <summary>Angle threshold within which corners are enabled. </summary>
        public float cornerAngleThreshold
        {
            get { return m_CornerAngleThreshold; }
            set { m_CornerAngleThreshold = value; }
        }

        /// <summary>Auto update colliders on any change to SpriteShape geometry. </summary>
        public bool autoUpdateCollider
        {
            get { return m_UpdateCollider; }
            set { m_UpdateCollider = value; }
        }

        /// <summary>Optimize generated collider geometry. </summary>
        public bool optimizeCollider
        {
            get { return true; }
        }

        /// <summary>Optimize generated SpriteShape geometry. </summary>
        public bool optimizeGeometry
        {
            get { return true; }
        }

        /// <summary>Does this SpriteShapeController object has colliders ?</summary>
        public bool hasCollider
        {
            get { return (edgeCollider != null) || (polygonCollider != null); }
        }

        /// <summary>Spline object that has data to create the Bezier curve of this SpriteShape Controller. </summary>
        public Spline spline
        {
            get { return m_Spline; }
        }

        /// <summary>Scale the Bounding Box of SpriteShape Geometry so its not culled out when scripting or dyanmically modifying Splines. </summary>
        public float boundsScale
        {
            get { return m_BoundsScale; }
            set { m_BoundsScale = value; InitBounds(); }
        }

        /// <summary>SpriteShape Profile asset that contains information on how to generate/render SpriteShapes. </summary>
        public SpriteShape spriteShape
        {
            get { return m_SpriteShape; }
            set { m_SpriteShape = value; }
        }

        /// <summary>EdgeCollider2D component attached to this Object.</summary>
        public EdgeCollider2D edgeCollider
        {
            get
            {
                if (!m_EdgeCollider2D)
                {
                    bool b = TryGetComponent(typeof(EdgeCollider2D), out Component comp);
                    m_EdgeCollider2D = b ? (comp as EdgeCollider2D) : null;
                }
                return m_EdgeCollider2D;
            }
        }

        /// <summary>PolygonCollider2D component attached to this Object.</summary>
        public PolygonCollider2D polygonCollider
        {
            get
            {
                if (!m_PolygonCollider2D)
                {
                    bool b = TryGetComponent(typeof(PolygonCollider2D), out Component comp);
                    m_PolygonCollider2D = b ? (comp as PolygonCollider2D) : null;
                }
                return m_PolygonCollider2D;
            }
        }

        /// <summary>SpriteShapeRenderer component of this Object. </summary>
        public SpriteShapeRenderer spriteShapeRenderer
        {
            get
            {
                if (!m_SpriteShapeRenderer)
                    m_SpriteShapeRenderer = GetComponent<SpriteShapeRenderer>();
                return m_SpriteShapeRenderer;
            }
        }

        internal bool updateShadow
        {
            get { return m_UpdateShadow; }
            set { m_UpdateShadow = value; }
        }

        internal int shadowDetail
        {
            get { return m_ShadowDetail; }
            set { m_ShadowDetail = value; }
        }

        internal float shadowOffset
        {
            get { return m_ShadowOffset; }
            set { m_ShadowOffset = value; }
        }

        internal List<Vector2> shadowSegment
        {
            get { return m_ShadowSegment; }
        }

        internal NativeArray<SpriteShapeGeneratorStats> stats
        {
            get
            {
                if (!m_Statistics.IsCreated)
                    m_Statistics = new NativeArray<SpriteShapeGeneratorStats>(1, Allocator.Persistent);
                return m_Statistics;
            }
        }

#endregion

#region EventHandles.

        void DisposeInternal()
        {
            m_JobHandle.Complete();
            if (m_ColliderData.IsCreated)
                m_ColliderData.Dispose();
            if (m_ShadowData.IsCreated)
                m_ShadowData.Dispose();
            if (m_TangentData.IsCreated)
                m_TangentData.Dispose();
            if (m_Statistics.IsCreated)
                m_Statistics.Dispose();
        }

        void OnApplicationQuit()
        {
            DisposeInternal();
        }

        void OnEnable()
        {
            m_DynamicOcclusionOverriden = true;
            m_DynamicOcclusionLocal = spriteShapeRenderer.allowOcclusionWhenDynamic;
            spriteShapeRenderer.allowOcclusionWhenDynamic = false;
            InitBounds();
            UpdateSpriteData();
        }

        void OnDisable()
        {
            UpdateGeometryCache();
            DisposeInternal();
        }

        void OnDestroy()
        {

        }

        void Reset()
        {
            m_SplineDetail = (int)QualityDetail.High;
            m_AdaptiveUV = true;
            m_StretchUV = false;
            m_FillPixelPerUnit = 100f;

            m_ColliderDetail = (int)QualityDetail.High;
            m_ShadowDetail = (int)QualityDetail.High;
            m_StretchTiling = 1.0f;
            m_WorldSpaceUV = false;
            m_CornerAngleThreshold = 30.0f;
            m_ColliderOffset = 0;
            m_ShadowOffset = 0.5f;
            m_UpdateCollider = true;
            m_EnableTangents = false;

            spline.Clear();
            spline.InsertPointAt(0, Vector2.left + Vector2.down);
            spline.InsertPointAt(1, Vector2.left + Vector2.up);
            spline.InsertPointAt(2, Vector2.right + Vector2.up);
            spline.InsertPointAt(3, Vector2.right + Vector2.down);
        }

        static void SmartDestroy(UnityEngine.Object o)
        {
            if (o == null)
                return;

#if UNITY_EDITOR
            if (!Application.isPlaying)
                DestroyImmediate(o);
            else
#endif
                Destroy(o);
        }

#endregion

#region HashAndDataCheck

        internal Bounds InitBounds()
        {
            var pointCount = spline.GetPointCount();
            if (pointCount > 1)
            {
                Bounds bounds = new Bounds(spline.GetPosition(0), Vector3.zero);
                for (int i = 1; i < pointCount; ++i)
                    bounds.Encapsulate(spline.GetPosition(i));
                bounds.size = bounds.size * m_BoundsScale;
                bounds.Encapsulate(spriteShapeRenderer.localBounds);
                spriteShapeRenderer.SetLocalAABB(bounds);
                return bounds;
            }
            return new Bounds();
        }

        /// <summary>
        /// Refresh SpriteShape Hash so its force generated again on the next frame if its visible.
        /// </summary>
        public void RefreshSpriteShape()
        {
            m_ActiveSplineHash = 0;
        }

        // Ensure Neighbor points are not too close to each other.
        bool ValidateSpline()
        {
            int pointCount = spline.GetPointCount();
            if (pointCount < 2)
                return false;
            for (int i = 0; i < pointCount - 1; ++i)
            {
                var vec = spline.GetPosition(i) - spline.GetPosition(i + 1);
                if (vec.sqrMagnitude < s_DistanceTolerance)
                {
                    Debug.LogWarningFormat(gameObject, "[SpriteShape] Control points {0} & {1} are too close. SpriteShape will not be generated for < {2} >.", i, i + 1, gameObject.name);
                    return false;
                }
            }
            return true;
        }

        // Ensure SpriteShape is valid if not
        bool ValidateSpriteShapeTexture()
        {
            bool valid = false;

            // Check if SpriteShape Profile is valid.
            if (spriteShape != null)
            {
                // Open ended and no valid Sprites set. Check if it has a valid fill texture.
                if (!spline.isOpenEnded)
                {
                    valid = (spriteShape.fillTexture != null);
                }
            }
            else
            {
                // Warn that no SpriteShape is set.
                Debug.LogWarningFormat(gameObject, "[SpriteShape] A valid SpriteShape profile has not been set for gameObject < {0} >.", gameObject.name);
#if UNITY_EDITOR
                // We allow null SpriteShape for rapid prototyping in Editor.
                valid = true;
#endif
            }
            return valid;
        }

        internal bool ValidateUTess2D()
        {
            bool uTess2D = m_UTess2D;
            // Check for all properties that can create overlaps/intersections.
            if (m_UTess2D && null != spriteShape)
            {
                uTess2D = (spriteShape.fillOffset == 0);
            }
            return uTess2D && !m_TessellationNeedsFallback;
        }

        bool HasSpriteShapeChanged()
        {
            bool changed = (m_ActiveSpriteShape != spriteShape);
            if (changed)
                m_ActiveSpriteShape = spriteShape;
            return changed;
        }

        bool HasSpriteShapeDataChanged()
        {
            bool updateSprites = HasSpriteShapeChanged();
            if (spriteShape)
            {
                var hashCode = SpriteShape.GetSpriteShapeHashCode(spriteShape);
                if (spriteShapeHashCode != hashCode)
                {
                    m_ActiveSpriteShapeHash = hashCode;
                    updateSprites = true;
                }
            }
            return updateSprites;
        }

        int GetCustomScriptHashCode()
        {

            int hashCode = 0;

            unchecked
            {
                hashCode = (int)2166136261 ^ spriteShapeCreator.GetVersion();
                foreach (var mod in m_Modifiers)
                    if (null != mod)
                        hashCode = hashCode * 16777619 ^ mod.GetVersion();
            }

            return hashCode;

        }

        bool HasSplineDataChanged()
        {
            unchecked
            {
                // Spline.
                int hashCode = (int)2166136261 ^ spline.GetHashCode();

                // Local Stuff.
                hashCode = hashCode * 16777619 ^ (m_UTess2D ? 1 : 0);
                hashCode = hashCode * 16777619 ^ (m_WorldSpaceUV ? 1 : 0);
                hashCode = hashCode * 16777619 ^ (m_EnableTangents ? 1 : 0);
                hashCode = hashCode * 16777619 ^ (m_GeometryCached ? 1 : 0);
                hashCode = hashCode * 16777619 ^ (m_UpdateShadow ? 1 : 0);
                hashCode = hashCode * 16777619 ^ (m_UpdateCollider ? 1 : 0);
                hashCode = hashCode * 16777619 ^ (m_StretchTiling.GetHashCode());
                hashCode = hashCode * 16777619 ^ (m_ColliderOffset.GetHashCode());
                hashCode = hashCode * 16777619 ^ (m_ColliderDetail.GetHashCode());
                hashCode = hashCode * 16777619 ^ (m_ShadowOffset.GetHashCode());
                hashCode = hashCode * 16777619 ^ (m_ShadowDetail.GetHashCode());
                hashCode = hashCode * 16777619 ^ (GetCustomScriptHashCode());
                hashCode = hashCode * 16777619 ^ (edgeCollider == null ? 0 : 1);
                hashCode = hashCode * 16777619 ^ (polygonCollider == null ? 0 : 1);

                if (splineHashCode != hashCode)
                {
                    m_ActiveSplineHash = hashCode;
                    return true;
                }
            }
            return false;
        }

        void OnBecameInvisible()
        {
            InitBounds();
        }

        void LateUpdate()
        {
            BakeCollider();
        }

        void OnWillRenderObject()
        {
            BakeMesh();
        }

        /// <summary>
        /// Generate geometry on a Job.
        /// </summary>
        /// <returns>JobHandle for the SpriteShape geometry generation job.</returns>
        public JobHandle BakeMesh()
        {
            JobHandle jobHandle = default;

#if !UNITY_EDITOR
            if (spriteShapeGeometryCache)
            {
                // If SpriteShapeGeometry has already been uploaded, don't bother checking further.
                if (0 != m_ActiveSplineHash && 0 != spriteShapeGeometryCache.maxArrayCount)
                    return jobHandle;
            }
#endif

            bool valid = ValidateSpline();

            if (valid)
            {
                bool splineChanged = HasSplineDataChanged();
                bool spriteShapeChanged = HasSpriteShapeDataChanged();
                bool spriteShapeParametersChanged = UpdateSpriteShapeParameters();

                if (splineChanged || spriteShapeChanged || spriteShapeParametersChanged || m_TessellationNeedsFallback)
                {
                    if (spriteShapeChanged)
                    {
                        UpdateSpriteData();
                    }
                    jobHandle = ScheduleBake();

#if UNITY_EDITOR
                    UpdateGeometryCache();
#endif
                }

            }
            return jobHandle;
        }

#endregion

#region UpdateData

        /// <summary>
        /// Update Cache.
        /// </summary>
        internal void UpdateGeometryCache()
        {
            if (spriteShapeGeometryCache && geometryCached)
            {
                m_JobHandle.Complete();
                spriteShapeGeometryCache.UpdateGeometryCache();
            }
        }

        /// <summary>
        /// Force update SpriteShape parameters.
        /// </summary>
        /// <returns>Returns true if there are changes</returns>
        public bool UpdateSpriteShapeParameters()
        {
            bool carpet = !spline.isOpenEnded;
            bool smartSprite = true;
            bool adaptiveUV = m_AdaptiveUV;
            bool stretchUV = m_StretchUV;
            bool spriteBorders = false;
            uint fillScale = 0;
            uint splineDetail = (uint)m_SplineDetail;
            float borderPivot = 0f;
            float angleThreshold = (m_CornerAngleThreshold >= 0 && m_CornerAngleThreshold < 90) ? m_CornerAngleThreshold : 89.9999f;
            Texture2D fillTexture = null;
            Matrix4x4 transformMatrix = Matrix4x4.identity;

            if (spriteShape)
            {
                if (worldSpaceUVs)
                    transformMatrix = transform.localToWorldMatrix;

                fillTexture = spriteShape.fillTexture;
                fillScale = stretchUV ? (uint)stretchTiling : (uint)fillPixelsPerUnit;
                borderPivot = spriteShape.fillOffset;
                spriteBorders = spriteShape.useSpriteBorders;
                // If Corners are enabled, set smart-sprite to false.
                if (spriteShape.cornerSprites.Count > 0)
                    smartSprite = false;
            }
            else
            {
#if UNITY_EDITOR
                fillTexture = UnityEditor.EditorGUIUtility.whiteTexture;
                fillScale = 100;
#endif
            }

            bool changed = m_ActiveShapeParameters.adaptiveUV != adaptiveUV ||
                m_ActiveShapeParameters.angleThreshold != angleThreshold ||
                m_ActiveShapeParameters.borderPivot != borderPivot ||
                m_ActiveShapeParameters.carpet != carpet ||
                m_ActiveShapeParameters.fillScale != fillScale ||
                m_ActiveShapeParameters.fillTexture != fillTexture ||
                m_ActiveShapeParameters.smartSprite != smartSprite ||
                m_ActiveShapeParameters.splineDetail != splineDetail ||
                m_ActiveShapeParameters.spriteBorders != spriteBorders ||
                m_ActiveShapeParameters.transform != transformMatrix ||
                m_ActiveShapeParameters.stretchUV != stretchUV;

            m_ActiveShapeParameters.adaptiveUV = adaptiveUV;
            m_ActiveShapeParameters.stretchUV = stretchUV;
            m_ActiveShapeParameters.angleThreshold = angleThreshold;
            m_ActiveShapeParameters.borderPivot = borderPivot;
            m_ActiveShapeParameters.carpet = carpet;
            m_ActiveShapeParameters.fillScale = fillScale;
            m_ActiveShapeParameters.fillTexture = fillTexture;
            m_ActiveShapeParameters.smartSprite = smartSprite;
            m_ActiveShapeParameters.splineDetail = splineDetail;
            m_ActiveShapeParameters.spriteBorders = spriteBorders;
            m_ActiveShapeParameters.transform = transformMatrix;

            return changed;
        }

        void UpdateSpriteData()
        {
            if (spriteShape)
            {
                List<Sprite> edgeSpriteList = new List<Sprite>();
                List<Sprite> cornerSpriteList = new List<Sprite>();
                List<AngleRangeInfo> angleRangeInfoList = new List<AngleRangeInfo>();

                List<AngleRange> sortedAngleRanges = new List<AngleRange>(spriteShape.angleRanges);
                sortedAngleRanges.Sort((a, b) => a.order.CompareTo(b.order));

                for (int i = 0; i < sortedAngleRanges.Count; i++)
                {
                    bool validSpritesFound = false;
                    AngleRange angleRange = sortedAngleRanges[i];
                    foreach (Sprite edgeSprite in angleRange.sprites)
                    {
                        if (edgeSprite != null)
                        {
                            validSpritesFound = true;
                            break;
                        }
                    }

                    if (validSpritesFound)
                    {
                        AngleRangeInfo angleRangeInfo = new AngleRangeInfo();
                        angleRangeInfo.start = angleRange.start;
                        angleRangeInfo.end = angleRange.end;
                        angleRangeInfo.order = (uint)angleRange.order;
                        List<int> spriteIndices = new List<int>();
                        foreach (Sprite edgeSprite in angleRange.sprites)
                        {
                            edgeSpriteList.Add(edgeSprite);
                            spriteIndices.Add(edgeSpriteList.Count - 1);
                        }
                        angleRangeInfo.sprites = spriteIndices.ToArray();
                        angleRangeInfoList.Add(angleRangeInfo);
                    }
                }

                bool validCornerSpritesFound = false;
                foreach (CornerSprite cornerSprite in spriteShape.cornerSprites)
                {
                    if (cornerSprite.sprites[0] != null)
                    {
                        validCornerSpritesFound = true;
                        break;
                    }
                }

                if (validCornerSpritesFound)
                {
                    for (int i = 0; i < spriteShape.cornerSprites.Count; i++)
                    {
                        CornerSprite cornerSprite = spriteShape.cornerSprites[i];
                        cornerSpriteList.Add(cornerSprite.sprites[0]);
                    }
                }

                m_EdgeSpriteArray = edgeSpriteList.ToArray();
                m_CornerSpriteArray = cornerSpriteList.ToArray();
                m_AngleRangeInfoArray = angleRangeInfoList.ToArray();

                List<Sprite> spriteList = new List<Sprite>();
                spriteList.AddRange(m_EdgeSpriteArray);
                spriteList.AddRange(m_CornerSpriteArray);
                m_SpriteArray = spriteList.ToArray();
            }
            else
            {
                m_SpriteArray = new Sprite[0];
                m_EdgeSpriteArray = new Sprite[0];
                m_CornerSpriteArray = new Sprite[0];
                m_AngleRangeInfoArray = new AngleRangeInfo[0];
            }
        }

        internal NativeArray<ShapeControlPoint> GetShapeControlPoints()
        {
            int pointCount = spline.GetPointCount();
            NativeArray<ShapeControlPoint> shapePoints = new NativeArray<ShapeControlPoint>(pointCount, Allocator.Temp);
            for (int i = 0; i < pointCount; ++i)
            {
                ShapeControlPoint shapeControlPoint;
                shapeControlPoint.position = spline.GetPosition(i);
                shapeControlPoint.leftTangent = spline.GetLeftTangent(i);
                shapeControlPoint.rightTangent = spline.GetRightTangent(i);
                shapeControlPoint.mode = (int)spline.GetTangentMode(i);
                shapePoints[i] = shapeControlPoint;
            }
            return shapePoints;
        }

        internal NativeArray<SplinePointMetaData> GetSplinePointMetaData()
        {
            int pointCount = spline.GetPointCount();
            NativeArray<SplinePointMetaData> shapeMetaData = new NativeArray<SplinePointMetaData>(pointCount, Allocator.Temp);
            for (int i = 0; i < pointCount; ++i)
            {
                SplinePointMetaData metaData;
                metaData.height = m_Spline.GetHeight(i);
                metaData.spriteIndex = (uint)m_Spline.GetSpriteIndex(i);
                metaData.cornerMode = (int)m_Spline.GetCornerMode(i);
                shapeMetaData[i] = metaData;
            }
            return shapeMetaData;
        }

        internal int CalculateMaxArrayCount(NativeArray<ShapeControlPoint> shapePoints)
        {
            int maxVertexCount = 1024 * 64;
            bool hasSprites = false;
            float smallestWidth = 99999.0f;

            if (null != spriteArray)
            {
                foreach (var sprite in m_SpriteArray)
                {
                    if (sprite != null)
                    {
                        hasSprites = true;
                        float pixelWidth = BezierUtility.GetSpritePixelWidth(sprite);
                        smallestWidth = (smallestWidth > pixelWidth) ? pixelWidth : smallestWidth;
                    }
                }
            }

            // Approximate vertex Array Count. Include Corners and Wide Sprites into account.
            float smallestSegment = smallestWidth;
            float shapeLength = BezierUtility.BezierLength(shapePoints, splineDetail, ref smallestSegment) * 4.0f;
            int adjustShape = shapePoints.Length * 5 * splineDetail;
            int adjustWidth = hasSprites ? ((int)(shapeLength / smallestSegment) * splineDetail) + adjustShape : 0;
            adjustShape = optimizeGeometry ? (adjustShape) : (adjustShape * 2);
            adjustShape = ValidateSpriteShapeTexture() ? adjustShape : 0;
            maxArrayCount = adjustShape + adjustWidth;
            maxArrayCount = math.min(maxArrayCount, maxVertexCount);
            return maxArrayCount;
        }

#endregion

#region ScheduleAndGenerate

        unsafe JobHandle ScheduleBake()
        {
            JobHandle jobHandle = default;

            bool staticUpload = Application.isPlaying;
#if !UNITY_EDITOR
            staticUpload = true;
#endif
            if (staticUpload && geometryCached)
            {
                if (spriteShapeGeometryCache)
                    if (spriteShapeGeometryCache.maxArrayCount != 0)
                        return spriteShapeGeometryCache.Upload(spriteShapeRenderer, this);
            }
            maxArrayCount = spriteShapeCreator.GetVertexArrayCount(this);

            if (maxArrayCount > 0 && enabled)
            {
                // Complate previos
                m_JobHandle.Complete();

                // Collider Data
                if (m_ColliderData.IsCreated)
                    m_ColliderData.Dispose();
                m_ColliderData = new NativeArray<float2>(maxArrayCount, Allocator.Persistent);
                if (m_ShadowData.IsCreated)
                    m_ShadowData.Dispose();
                m_ShadowData = new NativeArray<float2>(maxArrayCount, Allocator.Persistent);

                // Tangent Data
                if (!m_TangentData.IsCreated)
                    m_TangentData = new NativeArray<Vector4>(1, Allocator.Persistent);

                NativeArray<ushort> indexArray;
                NativeSlice<Vector3> posArray;
                NativeSlice<Vector2> uv0Array;
                NativeArray<SpriteShapeSegment> geomArray = spriteShapeRenderer.GetSegments(spline.GetPointCount() * 8);
                NativeSlice<Vector4> tanArray = new NativeSlice<Vector4>(m_TangentData);

                if (m_EnableTangents)
                {
                    spriteShapeRenderer.GetChannels(maxArrayCount, out indexArray, out posArray, out uv0Array, out tanArray);
                }
                else
                {
                    spriteShapeRenderer.GetChannels(maxArrayCount, out indexArray, out posArray, out uv0Array);
                }

                m_JobHandle = jobHandle = spriteShapeCreator.MakeCreatorJob(this, indexArray, posArray, uv0Array, tanArray, geomArray, m_ColliderData);
                foreach (var geomMod in m_Modifiers)
                    if (null != geomMod)
                        m_JobHandle = geomMod.MakeModifierJob(m_JobHandle, this, indexArray, posArray, uv0Array, tanArray, geomArray, m_ColliderData);

                // Prepare Renderer.
                spriteShapeRenderer.Prepare(m_JobHandle, m_ActiveShapeParameters, m_SpriteArray);
                jobHandle = m_JobHandle;
                m_TessellationNeedsFallback = false;

#if UNITY_EDITOR
                if (spriteShapeGeometryCache && geometryCached)
                    spriteShapeGeometryCache.SetGeometryCache(maxArrayCount, posArray, uv0Array, tanArray, indexArray, geomArray);
#endif

                JobHandle.ScheduleBatchedJobs();
            }

            if (m_DynamicOcclusionOverriden)
            {
                spriteShapeRenderer.allowOcclusionWhenDynamic = m_DynamicOcclusionLocal;
                m_DynamicOcclusionOverriden = false;
            }
            return jobHandle;
        }

        internal void BakeShadow()
        {
            if (m_ShadowData.IsCreated)
            {
                if (updateShadow)
                {
                    int maxCount = short.MaxValue - 1;
                    float2 last = (float2)0;

                    m_ShadowSegment.Clear();
                    for (int i = 0; i < maxCount; ++i)
                    {
                        float2 now = m_ShadowData[i];
                        if (!math.any(last) && !math.any(now))
                        {
                            if ((i + 1) < maxCount)
                            {
                                float2 next = m_ShadowData[i + 1];
                                if (!math.any(next) && !math.any(next))
                                    break;
                            }
                            else
                                break;
                        }
                        m_ShadowSegment.Add(new Vector2(now.x, now.y));
                    }

#if UNITY_EDITOR
                    UnityEditor.SceneView.RepaintAll();
#endif
                }

                // Dispose Collider as its no longer needed.
                m_ShadowData.Dispose();
            }
        }

        /// <summary>
        /// Update Collider of this Object.
        /// </summary>
        public void BakeCollider()
        {
            // Previously this must be explicitly called if using BakeMesh.
            // But now we do it internally. BakeCollider_CanBeCalledMultipleTimesWithoutJobComplete
            m_JobHandle.Complete();
            BakeShadow();

            if (m_ColliderData.IsCreated)
            {
                if ((autoUpdateCollider && hasCollider))
                {
                    int maxCount = short.MaxValue - 1;
                    float2 last = (float2)0;

                    m_ColliderSegment.Clear();
                    for (int i = 0; i < maxCount; ++i)
                    {
                        float2 now = m_ColliderData[i];
                        if (!math.any(last) && !math.any(now))
                        {
                            if ((i+1) < maxCount)
                            {
                                float2 next = m_ColliderData[i+1];
                                if (!math.any(next) && !math.any(next))
                                    break;
                            }
                            else
                                break;
                        }
                        m_ColliderSegment.Add(new Vector2(now.x, now.y));
                    }

                    if (autoUpdateCollider)
                    {
                        if (edgeCollider != null)
                            edgeCollider.points = m_ColliderSegment.ToArray();
                        if (polygonCollider != null)
                            polygonCollider.points = m_ColliderSegment.ToArray();
                    }
#if UNITY_EDITOR
                    UnityEditor.SceneView.RepaintAll();
#endif
                }

                // Dispose Collider as its no longer needed.
                m_ColliderData.Dispose();

                // Print Once.
                if (m_Statistics.IsCreated)
                {
                    var stats = m_Statistics[0];
                    switch (stats.status)
                    {
                        case SpriteShapeGeneratorResult.ErrorNativeDataOverflow:
                            Debug.LogWarningFormat(gameObject, "NativeArray access not within range. Please submit a bug report.");
                            break;
                        case SpriteShapeGeneratorResult.ErrorSpritesTightPacked:
                            Debug.LogWarningFormat(gameObject, "Sprites used in SpriteShape profile must use FullRect.");
                            break;
                        case SpriteShapeGeneratorResult.ErrorSpritesWrongBorder:
                            Debug.LogWarningFormat(gameObject, "Sprites used in SpriteShape profile have invalid borders. Please check SpriteShape profile.");
                            break;
                        case SpriteShapeGeneratorResult.ErrorVertexLimitReached:
                            Debug.LogWarningFormat(gameObject, "Mesh data has reached Limits. Please try dividing shape into smaller blocks.");
                            break;
                        case SpriteShapeGeneratorResult.ErrorDefaultQuadCreated:
                            if (m_UTess2D)
                            {
                                m_TessellationNeedsFallback = true;
                                Debug.LogWarningFormat(gameObject, "Fill tessellation (C# Job) encountered errors. Please avoid overlaps or close points in the input Spline. Falling back to default generator.");
                                break;
                            }
                            Debug.LogWarningFormat(gameObject, "Fill tessellation encountered errors. Please avoid overlaps or close points in the input spline.");
                            break;
                    }
                }
            }
        }

        internal void BakeMeshForced()
        {
            if (spriteShapeRenderer != null)
            {
                var hasSplineChanged = HasSplineDataChanged();
                if (hasSplineChanged)
                {
                    BakeMesh();
                    Rendering.CommandBuffer rc = new Rendering.CommandBuffer();
                    rc.GetTemporaryRT(0, 256, 256, 0);
                    rc.SetRenderTarget(0);
                    rc.DrawRenderer(spriteShapeRenderer, spriteShapeRenderer.sharedMaterial);
                    rc.ReleaseTemporaryRT(0);
                    Graphics.ExecuteCommandBuffer(rc);
                }
            }
        }

#endregion

        internal void ForceShadowShapeUpdate(bool forceUpdate)
        {
            m_UpdateShadow = forceUpdate;
        }

        internal NativeArray<float2> GetShadowShapeData()
        {
            if (m_ShadowData.IsCreated)
            {
                JobHandle handle = BakeMesh();
                handle.Complete();
                BakeCollider();
            }

            NativeArray<float2> retNativeArray = new NativeArray<float2>(m_ShadowSegment.Count, Allocator.Temp);
            for (int i = 0; i < retNativeArray.Length; i++)
                retNativeArray[i] = m_ShadowSegment[i];

            return retNativeArray;
        }
    }
}
