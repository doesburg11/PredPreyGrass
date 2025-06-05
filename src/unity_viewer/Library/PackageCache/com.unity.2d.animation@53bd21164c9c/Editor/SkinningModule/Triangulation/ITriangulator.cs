using Unity.Collections;
using Unity.Mathematics;

namespace UnityEditor.U2D.Animation
{
    internal interface ITriangulator
    {
        void Triangulate(ref int2[] edges, ref float2[] vertices, out int[] indices);
        Unity.Jobs.JobHandle ScheduleTriangulate(in float2[] vertices, in int2[] edges, ref NativeArray<float2> outputVertices, ref NativeArray<int> outputIndices, ref NativeArray<int2> outputEdges, ref NativeArray<int4> result);
        void Tessellate(float minAngle, float maxAngle, float meshAreaFactor, float largestTriangleAreaFactor, float areaThreshold, int smoothIterations, ref float2[] vertices, ref int2[] edges, out int[] indices);
        Unity.Jobs.JobHandle ScheduleTessellate(float minAngle, float maxAngle, float meshAreaFactor, float largestTriangleAreaFactor, float areaThreshold, int smoothIterations, in float2[] vertices, in int2[] edges, ref NativeArray<float2> outputVertices, ref NativeArray<int> outputIndices, ref NativeArray<int2> outputEdges, ref NativeArray<int4> result);
    }
}
