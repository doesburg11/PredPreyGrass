using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;

namespace UnityEngine.U2D
{
    /// <summary>
    /// Custom Post Processing after geometry is generated. 
    /// </summary>
    public abstract class SpriteShapeGeometryModifier : ScriptableObject
    {
        /// <summary>
        /// Modify generated geometry or override custom geometry.
        /// </summary>
        /// <param name="generator">JobHandle of the Main Job. default if Override.</param>
        /// <param name="spriteShapeController">SpriteShapeController from where this function is invoked from. </param>
        /// <param name="indices">Indices of generated geometry. </param>
        /// <param name="positions">Position of vertices in generated geometry. </param>
        /// <param name="texCoords">Texture Coordinates of vertices in generated geometry. </param>
        /// <param name="tangents">Tangent of vertices in generated geometry. </param>
        /// <param name="segments">Submeshes in generated geometry. </param>
        /// <param name="colliderData">Points that define the path of Collider. </param>
        /// <returns>JobHandle for the allocated Job to modify Geometry.</returns>
        public abstract JobHandle MakeModifierJob(JobHandle generator, SpriteShapeController spriteShapeController, NativeArray<ushort> indices,
            NativeSlice<Vector3> positions, NativeSlice<Vector2> texCoords, NativeSlice<Vector4> tangents,
            NativeArray<SpriteShapeSegment> segments, NativeArray<float2> colliderData);


        /// <summary>
        /// Get Versioning so we can check if geometry needs to be generated.
        /// </summary>
        /// <returns>Version of Modifier.</returns>
        public virtual int GetVersion() => GetInstanceID();
    }
};