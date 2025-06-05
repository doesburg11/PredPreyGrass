using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;
using ModuleHandle = UnityEngine.U2D.Common.UTess.ModuleHandle;
using Unity.Jobs;

namespace UnityEditor.U2D.Animation
{

    [BurstCompile]
    internal struct TriangulateJob : IJob
    {
        // Input Dataset
        [DeallocateOnJobCompletion]
        internal NativeArray<float2> inputVertices;
        [DeallocateOnJobCompletion]
        internal NativeArray<int2> inputEdges;

        // Output Dataset.
        internal NativeArray<int> outputIndices;
        internal NativeArray<int2> outputEdges;
        internal NativeArray<float2> outputVertices;
        internal NativeArray<int4> result;

        public void Execute()
        {
            int outputVertexCount = 0, outputIndexCount = 0, outputEdgeCount = 0;
            ModuleHandle.Tessellate(Allocator.Temp, inputVertices, inputEdges, ref outputVertices, out outputVertexCount, ref outputIndices, out outputIndexCount, ref outputEdges, out outputEdgeCount, true);
            result[0] = new int4(outputVertexCount, outputIndexCount, outputEdgeCount, 0);
        }
    }

    [BurstCompile]
    internal struct TessellateJob : IJob
    {
        // Input Parameters.
        internal int refineIterations;
        internal int smoothIterations;
        internal float minAngle;
        internal float maxAngle;
        internal float meshArea;
        internal float targetArea;
        internal float largestTriangleAreaFactor;

        // Input Dataset
        [DeallocateOnJobCompletion]
        internal NativeArray<float2> inputVertices;
        [DeallocateOnJobCompletion]
        internal NativeArray<int2> inputEdges;

        // Output Dataset.
        internal NativeArray<int> outputIndices;
        internal NativeArray<int2> outputEdges;
        internal NativeArray<float2> outputVertices;
        internal NativeArray<int4> result;

        public void Execute()
        {
            int outputVertexCount = 0, outputIndexCount = 0, outputEdgeCount = 0;
            ModuleHandle.Subdivide(Allocator.Temp, inputVertices, inputEdges, ref outputVertices, ref outputVertexCount, ref outputIndices, ref outputIndexCount, ref outputEdges, ref outputEdgeCount, largestTriangleAreaFactor, targetArea, refineIterations, smoothIterations);
            result[0] = new int4(outputVertexCount, outputIndexCount, outputEdgeCount, 0);
        }
    }

    [BurstCompile]
    internal class TriangulationUtility
    {

        // Adjust Tolerance for Collinear Check.
        static readonly float k_CollinearTolerance = 0.0001f;

        [BurstCompile]
        static unsafe int ValidateCollinear(float2* points, int pointCount, float epsilon)
        {
            if (pointCount < 3)
                return 0;

            for (int i = 0; i < pointCount - 2; ++i)
            {
                double2 a = points[i];
                double2 b = points[i + 1];
                double2 c = points[i + 2];

                // Just check area of triangle and see if its non-zero.
                var x = math.abs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
                if (x > epsilon)
                    return 1;
            }

            return 0;
        }

        [BurstCompile]
        static unsafe void TessellateBurst(Allocator allocator, float2* points, int pointCount, int2* edges, int edgeCount, float2* outVertices, int* outIndices, int2* outEdges, int arrayCount, int3* result)
        {

            var _edges = new NativeArray<int2>(edgeCount, allocator);
            for (int i = 0; i < _edges.Length; ++i)
                _edges[i] = edges[i];

            var _points = new NativeArray<float2>(pointCount, allocator);
            for (int i = 0; i < _points.Length; ++i)
                _points[i] = points[i];

            var _outIndices = new NativeArray<int>(arrayCount, allocator);
            var _outEdges = new NativeArray<int2>(arrayCount, allocator);
            var _outVertices = new NativeArray<float2>(arrayCount, allocator);

            var outEdgeCount = 0;
            var outIndexCount = 0;
            var outVertexCount = 0;

            var check = ValidateCollinear((float2*)_points.GetUnsafeReadOnlyPtr(), pointCount, k_CollinearTolerance);
            if (0 != check)
                ModuleHandle.Tessellate(allocator, in _points, in _edges, ref _outVertices, out outVertexCount, ref _outIndices, out outIndexCount, ref _outEdges, out outEdgeCount, true);

            for (var i = 0; i < outEdgeCount; ++i)
                outEdges[i] = _outEdges[i];
            for (var i = 0; i < outIndexCount; ++i)
                outIndices[i] = _outIndices[i];
            for (var i = 0; i < outVertexCount; ++i)
                outVertices[i] = _outVertices[i];

            result->x = outVertexCount;
            result->y = outIndexCount;
            result->z = outEdgeCount;

            _outVertices.Dispose();
            _outEdges.Dispose();
            _outIndices.Dispose();
            _points.Dispose();
            _edges.Dispose();

        }

        [BurstCompile]
        static unsafe void SubdivideBurst(Allocator allocator, float2* points, int pointCount, int2* edges, int edgeCount, float2* outVertices, int* outIndices, int2* outEdges, int arrayCount, float areaFactor, float areaThreshold, int refineIterations, int smoothenIterations, int3* result)
        {
            var _edges = new NativeArray<int2>(edgeCount, allocator);
            for (int i = 0; i < _edges.Length; ++i)
                _edges[i] = edges[i];

            var _points = new NativeArray<float2>(pointCount, allocator);
            for (int i = 0; i < _points.Length; ++i)
                _points[i] = points[i];

            var _outIndices = new NativeArray<int>(arrayCount, allocator);
            var _outEdges = new NativeArray<int2>(arrayCount, allocator);
            var _outVertices = new NativeArray<float2>(arrayCount, allocator);
            var outEdgeCount = 0;
            var outIndexCount = 0;
            var outVertexCount = 0;

            ModuleHandle.Subdivide(allocator, _points, _edges, ref _outVertices, ref outVertexCount, ref _outIndices, ref outIndexCount, ref _outEdges, ref outEdgeCount, areaFactor, areaThreshold, refineIterations, smoothenIterations);

            for (var i = 0; i < outEdgeCount; ++i)
                outEdges[i] = _outEdges[i];
            for (var i = 0; i < outIndexCount; ++i)
                outIndices[i] = _outIndices[i];
            for (var i = 0; i < outVertexCount; ++i)
                outVertices[i] = _outVertices[i];

            result->x = outVertexCount;
            result->y = outIndexCount;
            result->z = outEdgeCount;

            _outVertices.Dispose();
            _outEdges.Dispose();
            _outIndices.Dispose();
            _points.Dispose();
            _edges.Dispose();
        }

        static bool TessellateSafe(in NativeArray<float2> points, in NativeArray<int2> edges, ref NativeArray<float2> outVertices, ref int outVertexCount, ref NativeArray<int> outIndices, ref int outIndexCount, ref NativeArray<int2> outEdges, ref int outEdgeCount)
        {
            unsafe
            {
                var check = ValidateCollinear((float2*)points.GetUnsafeReadOnlyPtr(), points.Length, k_CollinearTolerance);
                if (0 == check)
                    return false;
            }

            try
            {
                ModuleHandle.Tessellate(Allocator.Persistent, in points, in edges, ref outVertices, out outVertexCount, ref outIndices, out outIndexCount, ref outEdges, out outEdgeCount, true);
            }
            catch (Exception)
            {
                return false;
            }
            return true;
        }
        static bool SubdivideSafe(NativeArray<float2> points, NativeArray<int2> edges, ref NativeArray<float2> outVertices, ref int outVertexCount, ref NativeArray<int> outIndices, ref int outIndexCount, ref NativeArray<int2> outEdges, ref int outEdgeCount, float areaFactor, float areaThreshold, int refineIterations, int smoothenIterations)
        {
            try
            {
                ModuleHandle.Subdivide(Allocator.Persistent, points, edges, ref outVertices, ref outVertexCount, ref outIndices, ref outIndexCount, ref outEdges, ref outEdgeCount, areaFactor, areaThreshold, refineIterations, smoothenIterations);
            }
            catch (Exception)
            {
                return false;
            }
            return true;
        }

        internal static void Quad(IList<Vector2> vertices, IList<Vector2Int> edges, IList<int> indices, Allocator allocator)
        {
            if (vertices.Count < 3)
                return;

            var points = new NativeArray<float2>(vertices.Count, allocator);
            for (var i = 0; i < vertices.Count; ++i)
                points[i] = vertices[i];

            var arrayCount = vertices.Count * vertices.Count * 4;
            int vertexCount = 0, indexCount = 0, edgeCount = 0;
            var outputIndices = new NativeArray<int>(arrayCount, allocator);
            var outputEdges = new NativeArray<int2>(arrayCount, allocator);
            var outputVertices = new NativeArray<float2>(arrayCount, allocator);

            var fallback = new NativeArray<int2>(0, allocator);
            TessellateSafe(in points, in fallback, ref outputVertices, ref vertexCount, ref outputIndices,
                ref indexCount, ref outputEdges, ref edgeCount);
            fallback.Dispose();

            vertices.Clear();
            for (var i = 0; i < vertexCount; ++i)
                vertices.Add(outputVertices[i]);
            indices.Clear();
            for (var i = 0; i < indexCount; ++i)
                indices.Add(outputIndices[i]);
            edges.Clear();
            for (var i = 0; i < edgeCount; ++i)
                edges.Add(new Vector2Int(outputEdges[i].x, outputEdges[i].y));

            outputEdges.Dispose();
            outputIndices.Dispose();
            outputVertices.Dispose();
            points.Dispose();
        }

        internal static void Triangulate(ref int2[] edges, ref float2[] vertices, out int[] indices, Allocator allocator)
        {
            if (vertices.Length < 3)
            {
                indices = new int[0];
                return;
            }

            var points = new NativeArray<float2>(vertices, allocator);
            var inputEdges = new NativeArray<int2>(edges, allocator);

            var arrayCount = vertices.Length * vertices.Length * 4;
            int vertexCount = 0, indexCount = 0, edgeCount = 0;
            var outputIndices = new NativeArray<int>(arrayCount, allocator);
            var outputEdges = new NativeArray<int2>(arrayCount, allocator);
            var outputResult = new NativeArray<int3>(1, allocator);
            var outputVertices = new NativeArray<float2>(arrayCount, allocator);

            unsafe
            {
                TessellateBurst(allocator, (float2*)points.GetUnsafePtr(), points.Length, (int2*)inputEdges.GetUnsafePtr(), inputEdges.Length, (float2*)outputVertices.GetUnsafePtr(), (int*)outputIndices.GetUnsafePtr(), (int2*)outputEdges.GetUnsafePtr(), arrayCount, (int3*)outputResult.GetUnsafePtr());
                vertexCount = outputResult[0].x;
                indexCount = outputResult[0].y;
                edgeCount = outputResult[0].z;
            }
            // Fallback on numerical precision errors.
            if (vertexCount <= 8 || indexCount == 0)
                TessellateSafe(in points, in inputEdges, ref outputVertices, ref vertexCount, ref outputIndices, ref indexCount, ref outputEdges, ref edgeCount);

            vertices = new float2[vertexCount];
            for (var i = 0; i < vertexCount; ++i)
                vertices[i] = outputVertices[i];
            indices = new int[indexCount];
            for (var i = 0; i < indexCount; ++i)
                indices[i] = outputIndices[i];
            edges = new int2[edgeCount];
            for (var i = 0; i < edgeCount; ++i)
                edges[i] = outputEdges[i];

            outputEdges.Dispose();
            outputResult.Dispose();
            outputIndices.Dispose();
            outputVertices.Dispose();
            inputEdges.Dispose();
            points.Dispose();
        }

        internal static bool TriangulateSafe(ref float2[] vertices, ref int2[] edges, out int[] indices)
        {
            indices = new int[0];

            if (vertices.Length < 3)
                return false;

            var points = new NativeArray<float2>(vertices, Allocator.Persistent);
            var inputEdges = new NativeArray<int2>(edges, Allocator.Persistent);

            var arrayCount = vertices.Length * vertices.Length * 4;
            int vertexCount = 0, indexCount = 0, edgeCount = 0;
            var outputIndices = new NativeArray<int>(arrayCount, Allocator.Persistent);
            var outputEdges = new NativeArray<int2>(arrayCount, Allocator.Persistent);
            var outputVertices = new NativeArray<float2>(arrayCount, Allocator.Persistent);
            var ok = TessellateSafe(in points, in inputEdges, ref outputVertices, ref vertexCount, ref outputIndices, ref indexCount, ref outputEdges, ref edgeCount);

            if (ok)
            {
                vertices = new float2[vertexCount];
                for (var i = 0; i < vertexCount; ++i)
                    vertices[i] = outputVertices[i];
                edges = new int2[edgeCount];
                for (var i = 0; i < edgeCount; ++i)
                    edges[i] = outputEdges[i];
                indices = new int[indexCount];
                for (var i = 0; i < indexCount; ++i)
                    indices[i] = outputIndices[i];
            }

            outputEdges.Dispose();
            outputIndices.Dispose();
            outputVertices.Dispose();
            inputEdges.Dispose();
            points.Dispose();
            return ok;
        }

        public static void Tessellate(float minAngle, float maxAngle, float meshAreaFactor, float largestTriangleAreaFactor, float targetArea, int refineIterations, int smoothenIterations, ref float2[] vertices, ref int2[] edges, out int[] indices, Allocator allocator)
        {
            indices = new int[0];

            if (vertices.Length < 3)
                return;

            largestTriangleAreaFactor = Mathf.Clamp01(largestTriangleAreaFactor);

            var points = new NativeArray<float2>(vertices.Length, allocator);
            for (var i = 0; i < vertices.Length; ++i)
                points[i] = vertices[i];
            var inputEdges = new NativeArray<int2>(edges.Length, allocator);
            for (var i = 0; i < edges.Length; ++i)
                inputEdges[i] = new int2(edges[i].x, edges[i].y);

            const int maxDataCount = 65536;
            int vertexCount = 0, indexCount = 0, edgeCount = 0;
            var outputIndices = new NativeArray<int>(maxDataCount, allocator);
            var outputEdges = new NativeArray<int2>(maxDataCount, allocator);
            var outputResult = new NativeArray<int3>(1, allocator);
            var outputVertices = new NativeArray<float2>(maxDataCount, allocator);

            unsafe
            {
                SubdivideBurst(allocator, (float2*)points.GetUnsafePtr(), points.Length, (int2*)inputEdges.GetUnsafePtr(), inputEdges.Length, (float2*)outputVertices.GetUnsafePtr(), (int*)outputIndices.GetUnsafePtr(), (int2*)outputEdges.GetUnsafePtr(), maxDataCount, largestTriangleAreaFactor, targetArea, refineIterations, smoothenIterations, (int3*)outputResult.GetUnsafePtr());
                vertexCount = outputResult[0].x;
                indexCount = outputResult[0].y;
                edgeCount = outputResult[0].z;
            }
            // Fallback on numerical precision errors.
            if (vertexCount <= 8)
                SubdivideSafe(points, inputEdges, ref outputVertices, ref vertexCount, ref outputIndices, ref indexCount, ref outputEdges, ref edgeCount, largestTriangleAreaFactor, targetArea, refineIterations, smoothenIterations);

            vertices = new float2[vertexCount];
            for (var i = 0; i < vertexCount; ++i)
                vertices[i] = outputVertices[i];
            edges = new int2[edgeCount];
            for (var i = 0; i < edgeCount; ++i)
                edges[i] = outputEdges[i];
            indices = new int[indexCount];
            for (var i = 0; i < indexCount; ++i)
                indices[i] = outputIndices[i];

            outputEdges.Dispose();
            outputResult.Dispose();
            outputIndices.Dispose();
            outputVertices.Dispose();
            inputEdges.Dispose();
            points.Dispose();
        }

        public static JobHandle ScheduleTriangulate(in float2[] vertices, in int2[] edges, ref NativeArray<float2> outputVertices, ref NativeArray<int2> outputEdges, ref NativeArray<int> outputIndices, ref NativeArray<int4> result)
        {
            if (vertices.Length < 3)
                return default(JobHandle);

            var inputVertices = new NativeArray<float2>(vertices.Length, Allocator.TempJob);
            for (var i = 0; i < vertices.Length; ++i)
                inputVertices[i] = vertices[i];
            var inputEdges = new NativeArray<int2>(edges.Length, Allocator.TempJob);
            for (var i = 0; i < edges.Length; ++i)
                inputEdges[i] = new int2(edges[i].x, edges[i].y);

            var tessAsJob = new TriangulateJob();
            tessAsJob.inputVertices = inputVertices;
            tessAsJob.inputEdges = inputEdges;
            tessAsJob.outputVertices = outputVertices;
            tessAsJob.outputIndices = outputIndices;
            tessAsJob.outputEdges = outputEdges;
            tessAsJob.result = result;
            return tessAsJob.Schedule();
        }

        public static JobHandle ScheduleTessellate(float minAngle, float maxAngle, float meshAreaFactor, float largestTriangleAreaFactor, float targetArea, int refineIterations, int smoothenIterations, in float2[] vertices, in int2[] edges, ref NativeArray<float2> outputVertices, ref NativeArray<int2> outputEdges, ref NativeArray<int> outputIndices, ref NativeArray<int4> result)
        {
            if (vertices.Length < 3)
                return default(JobHandle);

            largestTriangleAreaFactor = Mathf.Clamp01(largestTriangleAreaFactor);

            var inputVertices = new NativeArray<float2>(vertices.Length, Allocator.TempJob);
            for (var i = 0; i < vertices.Length; ++i)
                inputVertices[i] = vertices[i];
            var inputEdges = new NativeArray<int2>(edges.Length, Allocator.TempJob);
            for (var i = 0; i < edges.Length; ++i)
                inputEdges[i] = new int2(edges[i].x, edges[i].y);

            var tessAsJob = new TessellateJob();
            tessAsJob.minAngle = minAngle;
            tessAsJob.maxAngle = maxAngle;
            tessAsJob.meshArea = meshAreaFactor;
            tessAsJob.largestTriangleAreaFactor = largestTriangleAreaFactor;
            tessAsJob.targetArea = targetArea;
            tessAsJob.refineIterations = refineIterations;
            tessAsJob.smoothIterations = smoothenIterations;
            tessAsJob.inputVertices = inputVertices;
            tessAsJob.inputEdges = inputEdges;
            tessAsJob.outputVertices = outputVertices;
            tessAsJob.outputIndices = outputIndices;
            tessAsJob.outputEdges = outputEdges;
            tessAsJob.result = result;
            return tessAsJob.Schedule();
        }

        public static void TessellateSafe(float largestTriangleAreaFactor, float targetArea, int refineIterations, int smoothenIterations, ref float2[] vertices, ref int2[] edges, out int[] indices)
        {
            indices = new int[0];

            if (vertices.Length < 3)
                return;

            largestTriangleAreaFactor = Mathf.Clamp01(largestTriangleAreaFactor);

            var points = new NativeArray<float2>(vertices, Allocator.Persistent);
            var inputEdges = new NativeArray<int2>(edges, Allocator.Persistent);

            int vertexCount = 0, indexCount = 0, edgeCount = 0, maxDataCount = 65536;
            var outputVertices = new NativeArray<float2>(maxDataCount, Allocator.Persistent);
            var outputIndices = new NativeArray<int>(maxDataCount, Allocator.Persistent);
            var outputEdges = new NativeArray<int2>(maxDataCount, Allocator.Persistent);
            var ok = SubdivideSafe(points, inputEdges, ref outputVertices, ref vertexCount, ref outputIndices, ref indexCount, ref outputEdges, ref edgeCount, largestTriangleAreaFactor, targetArea, refineIterations, smoothenIterations) ;

            if (ok)
            {
                vertices = new float2[vertexCount];
                for (var i = 0; i < vertices.Length; ++i)
                    vertices[i] = outputVertices[i];
                indices = new int[indexCount];
                for (var i = 0; i < indices.Length; ++i)
                    indices[i] = outputIndices[i];
                edges = new int2[edgeCount];
                for (var i = 0; i < edges.Length; ++i)
                    edges[i] = outputEdges[i];
            }

            outputEdges.Dispose();
            outputIndices.Dispose();
            outputVertices.Dispose();
            inputEdges.Dispose();
            points.Dispose();
        }

        // Triangulate Bone Samplers.  todo: Burst it.
        internal static void TriangulateSamplers(in float2[] samplers, ref List<float2> triVertices, ref List<int> triIndices)
        {
            foreach(var v in samplers)
            {
                var vertexCount = triVertices.Count;

                for (var i = 0; i < triIndices.Count / 3; ++i)
                {
                    var i1 = triIndices[0 + (i * 3)];
                    var i2 = triIndices[1 + (i * 3)];
                    var i3 = triIndices[2 + (i * 3)];
                    var v1 = triVertices[i1];
                    var v2 = triVertices[i2];
                    var v3 = triVertices[i3];
                    var inside = ModuleHandle.IsInsideTriangle(v, v1, v2, v3);
                    if (inside)
                    {
                        triVertices.Add(v);
                        triIndices.Add(i1); triIndices.Add(i2); triIndices.Add(vertexCount);
                        triIndices.Add(i2); triIndices.Add(i3); triIndices.Add(vertexCount);
                        triIndices.Add(i3); triIndices.Add(i1); triIndices.Add(vertexCount);
                        break;
                    }
                }
            }
        }


        // Triangulate Skipped Original Points. These points are discarded during PlanarGrapg cleanup. But bbw only cares if these are part of any geometry. So just insert them. todo: Burst it.
        internal static void TriangulateInternal(in int[] internalIndices, in float2[] triVertices, ref List<int> triIndices)
        {
            var triangleCount = triIndices.Count / 3;

            foreach(var index in internalIndices)
            {
                var v = triVertices[index];
                for (var i = 0; i < triangleCount; ++i)
                {
                    var i1 = triIndices[0 + (i * 3)];
                    var i2 = triIndices[1 + (i * 3)];
                    var i3 = triIndices[2 + (i * 3)];
                    var v1 = triVertices[i1];
                    var v2 = triVertices[i2];
                    var v3 = triVertices[i3];
                    var c1 = (float)Math.Round(ModuleHandle.OrientFast(v1, v2, v), 2);
                    if (c1 == 0)
                    {
                        triIndices[0 + (i * 3)] = i1; triIndices[1 + (i * 3)] = index; triIndices[2 + (i * 3)] = i3;
                        triIndices.Add(index); triIndices.Add(i2); triIndices.Add(i3);
                    }
                    else
                    {
                        var c2 = (float)Math.Round(ModuleHandle.OrientFast(v2, v3, v), 2);
                        if (c2 == 0)
                        {
                            triIndices[0 + (i * 3)] = i2; triIndices[1 + (i * 3)] = index; triIndices[2 + (i * 3)] = i1;
                            triIndices.Add(index); triIndices.Add(i3); triIndices.Add(i1);
                        }
                        else
                        {
                            var c3 = (float)Math.Round(ModuleHandle.OrientFast(v3, v1, v), 2);
                            if (c3 == 0)
                            {
                                triIndices[0 + (i * 3)] = i3; triIndices[1 + (i * 3)] = index; triIndices[2 + (i * 3)] = i2;
                                triIndices.Add(index); triIndices.Add(i1); triIndices.Add(i2);
                            }
                        }
                    }
                }
            }
        }

    }
}
