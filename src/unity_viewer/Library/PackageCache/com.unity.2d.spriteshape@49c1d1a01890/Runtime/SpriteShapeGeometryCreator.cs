using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;

namespace UnityEngine.U2D
{
    /// <summary>
    /// Custom Post Processing after geometry is generated. 
    /// </summary>
    public abstract class SpriteShapeGeometryCreator : ScriptableObject
    {
        /// <summary>
        /// Get size of the vertices to be allocated for the Job. This is also used to determine the number of indices needed.
        /// Current implementaiton only allows 1 vertex to be mapped to 1 index thus the index array will have the same length as the vertex array.
        /// </summary>
        /// <param name="spriteShapeController">SpriteShapeController of the GameObject.</param>
        /// <returns>Size of the VertexData to be allocated</returns>
        public abstract int GetVertexArrayCount(SpriteShapeController spriteShapeController);

        /// <summary>
        /// Create SpriteShape Geometry.
        /// </summary>
        /// <param name="spriteShapeController">SpriteShapeController of the GameObject.</param>
        /// <param name="indices">Indices of generated geometry. Initialize to max Array count and contains default data. </param>
        /// <param name="positions">Position of vertices in generated geometry. Initialize to max Array count and contains default data. </param>
        /// <param name="texCoords">Texture Coordinates of vertices in generated geometry. Initialize to max Array count and contains default data. </param>
        /// <param name="tangents">Tangent of vertices in generated geometry. Initialize to max Array count and contains default data. </param>
        /// <param name="segments">Submeshes in generated geometry. Initialize to max Array count and contains default data. </param>
        /// <param name="colliderData">Points that define the path of Collider. </param>
        /// <returns>JobHandle for the allocated Job to generate Geometry.</returns>
        public abstract JobHandle MakeCreatorJob(SpriteShapeController spriteShapeController, NativeArray<ushort> indices,
            NativeSlice<Vector3> positions, NativeSlice<Vector2> texCoords, NativeSlice<Vector4> tangents,
            NativeArray<SpriteShapeSegment> segments, NativeArray<float2> colliderData);

        /// <summary>
        /// Get Versioning so we can check if geometry needs to be generated.
        /// </summary>
        /// <returns>Version of Generator.</returns>
        public virtual int GetVersion() => GetInstanceID();
    }
};
