using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal class Triangulator : ITriangulator
    {
        public void Triangulate(ref int2[] edges, ref float2[] vertices, out int[] indices)
        {
            TriangulationUtility.Triangulate(ref edges, ref vertices, out indices, Allocator.Persistent);
        }

        public Unity.Jobs.JobHandle ScheduleTriangulate(in float2[] vertices, in int2[] edges, ref NativeArray<float2> outputVertices, ref NativeArray<int> outputIndices, ref NativeArray<int2> outputEdges, ref NativeArray<int4> result)
        {
            return TriangulationUtility.ScheduleTriangulate(in vertices, in edges, ref outputVertices, ref outputEdges, ref outputIndices, ref result);
        }

        public void Tessellate(float minAngle, float maxAngle, float meshAreaFactor, float largestTriangleAreaFactor, float areaThreshold, int smoothIterations, ref float2[] vertices, ref int2[] edges, out int[] indices)
        {
            TriangulationUtility.Tessellate(minAngle, maxAngle, meshAreaFactor, largestTriangleAreaFactor, areaThreshold, 10, smoothIterations, ref vertices, ref edges, out indices, Allocator.Persistent);
        }

        public Unity.Jobs.JobHandle ScheduleTessellate(float minAngle, float maxAngle, float meshAreaFactor, float largestTriangleAreaFactor, float areaThreshold, int smoothIterations, in float2[] vertices, in int2[] edges, ref NativeArray<float2> outputVertices, ref NativeArray<int> outputIndices, ref NativeArray<int2> outputEdges, ref NativeArray<int4> result)
        {
            return TriangulationUtility.ScheduleTessellate(minAngle, maxAngle, meshAreaFactor, largestTriangleAreaFactor, areaThreshold, 10, smoothIterations, in vertices, in edges, ref outputVertices, ref outputEdges, ref outputIndices, ref result);
        }
    }
}
