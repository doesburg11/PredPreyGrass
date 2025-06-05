using System.Collections.Generic;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;

namespace UnityEngine.U2D
{

    internal class SpriteShapeDefaultCreator : SpriteShapeGeometryCreator
    {

        public override int GetVertexArrayCount(SpriteShapeController sc)
        {
            NativeArray<ShapeControlPoint> shapePoints = sc.GetShapeControlPoints();
            sc.CalculateMaxArrayCount(shapePoints);
            shapePoints.Dispose();
            return sc.maxArrayCount;
        }

        public override JobHandle MakeCreatorJob(SpriteShapeController sc,
            NativeArray<ushort> indices, NativeSlice<Vector3> positions, NativeSlice<Vector2> texCoords,
            NativeSlice<Vector4> tangents, NativeArray<SpriteShapeSegment> segments, NativeArray<float2> colliderData)
        {
            var uTess2D = sc.ValidateUTess2D();
            NativeArray<Bounds> bounds = sc.spriteShapeRenderer.GetBounds();
            var spriteShapeJob = new SpriteShapeGenerator()
            {
                m_Bounds = bounds, m_PosArray = positions, m_Uv0Array = texCoords, m_TanArray = tangents,
                m_GeomArray = segments, m_IndexArray = indices, m_ColliderPoints = colliderData, m_Stats = sc.stats, m_ShadowPoints = sc.shadowData
            };
            spriteShapeJob.generateCollider = SpriteShapeController.generateCollider;
            spriteShapeJob.generateGeometry = SpriteShapeController.generateGeometry;

            var shapePoints = sc.GetShapeControlPoints();
            var shapeMetaData = sc.GetSplinePointMetaData();
            spriteShapeJob.Prepare(sc, sc.spriteShapeParameters, sc.maxArrayCount, shapePoints, shapeMetaData,
                sc.angleRangeInfoArray, sc.edgeSpriteArray, sc.cornerSpriteArray, uTess2D);
            var jobHandle = spriteShapeJob.Schedule();
            shapePoints.Dispose();
            shapeMetaData.Dispose();
            return jobHandle;
        }

        static SpriteShapeDefaultCreator creator;

        internal static SpriteShapeDefaultCreator defaultInstance
        {
            get
            {
                if (null == creator)
                {
                    creator = ScriptableObject.CreateInstance<SpriteShapeDefaultCreator>();
                    creator.hideFlags = HideFlags.DontSave;
                }
                return creator;                
            }
        }


        /// <summary>
        /// Get Versioning so we can check if geometry needs to be generated.
        /// </summary>
        public override int GetVersion()
        {
            int hashCode = 0;            
            int versionHash = 1;
            unchecked
            {
                hashCode = (int)2166136261 ^ GetInstanceID();
                hashCode = hashCode * 16777619 ^ versionHash;
            }
            return hashCode;
        }

    }

}